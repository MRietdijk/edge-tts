"""Disk-backed audio cache using flat file + JSON index.

Stores TTS chunks (audio + metadata) per word in a binary file,
with a JSON index mapping words to their location in the file.
On cache hit, data is read from disk instead of making a network request.
"""

import json
import mmap
import os
import struct
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

from .typing import TTSChunk

class CacheInterface(ABC):
    @abstractmethod
    def get(self, word: str) -> Optional[list[TTSChunk]]:
        pass

    @abstractmethod
    def put(self, key: str, data: list[TTSChunk]) -> None:
        pass

def serialize_chunks(chunks: List[TTSChunk]) -> bytes:
    """
    Serialize a list of TTSChunk dicts into bytes.
    """
    parts = []
    # Number of chunks
    parts.append(struct.pack(">I", len(chunks)))

    for chunk in chunks:
        if chunk["type"] == "audio":
            # Header: just the type
            header = json.dumps({"type": "audio"}).encode("utf-8")
            audio_data = chunk["data"]
        else:
            # Metadata chunk: store everything except 'data' key
            header = json.dumps({
                k: v for k, v in chunk.items()
            }).encode("utf-8")
            audio_data = b""

        parts.append(struct.pack(">I", len(header)))
        parts.append(header)
        parts.append(struct.pack(">I", len(audio_data)))
        if audio_data:
            parts.append(audio_data)

    return b"".join(parts)

def deserialize_chunks(blob: bytes) -> List[TTSChunk]:
    """
    Deserialize bytes back into a list of TTSChunk dicts.
    """
    pos = 0
    (num_chunks,) = struct.unpack(">I", blob[pos:pos + 4])
    pos += 4

    chunks: List[TTSChunk] = []
    for _ in range(num_chunks):
        (header_len,) = struct.unpack(">I", blob[pos:pos + 4])
        pos += 4
        header = json.loads(blob[pos:pos + header_len])
        pos += header_len

        (audio_len,) = struct.unpack(">I", blob[pos:pos + 4])
        pos += 4

        if header["type"] == "audio":
            # Read audio bytes directly from the blob (which is an mmap slice)
            chunks.append({"type": "audio", "data": blob[pos:pos + audio_len]})
        else:
            chunks.append(header)

        pos += audio_len

    return chunks

class AudioCache(CacheInterface):
    def __init__(self, cache_dir: str = ".tts_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.data_path = os.path.join(cache_dir, "words.bin")
        self.index_path = os.path.join(cache_dir, "words.idx")

        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                self.index: Dict[str, Tuple[int, int]] = json.load(f)
        else:
            self.index = {}

        self.pre_allocation_size = 10000000 #10 megabytes in bytes
        self.allocation_pointer = 0
        self.file_size = 0

        self._mmap: mmap.mmap = self._construct_mmap()

    def _construct_mmap(self) -> mmap.mmap:
        file_size = os.path.getsize(self.data_path)
        fd = os.open(self.data_path, os.O_RDWR)
        self.file_size = file_size +\
            (self.pre_allocation_size if self.allocation_pointer == 0 else self.allocation_pointer)
        os.ftruncate(fd, self.file_size)

        new_mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_WRITE)
        os.close(fd)
        self._mmap = new_mmap
        return new_mmap
    
    def _invalidate_rebuild(self) -> mmap.mmap:
        self._mmap.close()
        new_map = self._construct_mmap()
        self.allocation_pointer = 0
        return new_map

    def get(self, word: str) -> Optional[List[TTSChunk]]:
        try:
            offset, length = self.index[word]
        except KeyError:
            return

        return deserialize_chunks(self._mmap[offset:offset + length])

    def put(self, key: str, data: List[TTSChunk]):
        if key in self.index:
            raise KeyError(f"Cache entry {key} already exists!")

        serialized = serialize_chunks(data)
        data_size = len(serialized)
        start = -self.pre_allocation_size + self.allocation_pointer
        end =  start + data_size

        if end >= 0:
            self._invalidate_rebuild()
            start = -self.pre_allocation_size + self.allocation_pointer
            end =  start + data_size

        self._mmap[start:end] = serialized
        self.index[key] = (self.file_size + start, data_size)
        self.allocation_pointer += data_size

    def save_index(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump(self.index, f)

    def cleanup(self) -> None:
        self._mmap.close()
        fd = os.open(self.data_path, os.O_RDWR)
        new_size = self.file_size - (self.pre_allocation_size - self.allocation_pointer)
        os.ftruncate(fd, new_size)
        os.close(fd)

class AudioCachePerWord(CacheInterface):
    def __init__(self) -> None:
        self.path: Path = Path('./.cache')
        os.makedirs(self.path, exist_ok=True)

    def get(self, word: str) -> Optional[List[TTSChunk]]:
        try:
            with open(self.path.joinpath(f"{str(word)}.bin"), "rb") as f:
                return deserialize_chunks(f.read())
        except FileNotFoundError:
            return
        
    def put(self, key: str, data: list[TTSChunk]) -> None:
        path = self.path.joinpath(f"{key}.bin")
        if os.path.isfile(path):
            raise KeyError(f"Cache file {path} already exists!")
        
        with open(path, "wb") as f:
            f.write(serialize_chunks(data))


class DictCache(CacheInterface):
    def __init__(self) -> None:
        self.cache = {}

    def get(self, word: str) -> Optional[List[TTSChunk]]:
        try:
            return self.cache[word]
        except KeyError:
            return
        
    def put(self, key: str, data: list[TTSChunk]) -> None:
        if key in self.cache:
            raise KeyError(f"Cache entry {key} already exists!")
        
        self.cache[key] = data

class CachePerWordDictCache(CacheInterface):
    def __init__(self) -> None:
        self.dict_cache = DictCache()
        self.word_cache = AudioCachePerWord()

    def get(self, word: str) -> Optional[List[TTSChunk]]:
        result = self.dict_cache.get(word)
        if result is not None: 
            return result
        
        result = self.word_cache.get(word)
        if result is None:
            return
        self.dict_cache.put(word, result)
        return result
    
    def put(self, key: str, data: List[TTSChunk]):
        self.word_cache.put(key, data)
        self.dict_cache.put(key, data)
