"""Disk-backed audio cache using flat file + JSON index.

Stores TTS chunks (audio + metadata) per word in a binary file,
with a JSON index mapping words to their location in the file.
On cache hit, data is read from disk instead of making a network request.
"""

import json
import mmap
import os
import struct
from typing import Dict, List, Optional

from .typing import TTSChunk


class AudioCache:
    """
    Persists TTS chunks to disk using a flat binary file + JSON index.

    Layout of the .bin file for each word:
        [4 bytes: number of chunks N]
        For each chunk:
            [4 bytes: length of JSON-serialized chunk header]
            [header bytes: JSON with type, offset, duration, text — no audio data]
            [4 bytes: length of audio data (0 if not an audio chunk)]
            [audio bytes]

    The .idx file maps: word (str) -> {"offset": int, "length": int}
    """

    def __init__(self, cache_dir: str = ".tts_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.data_path = os.path.join(cache_dir, "words.bin")
        self.index_path = os.path.join(cache_dir, "words.idx")

        # Load existing index or start fresh
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                self.index: Dict[str, Dict[str, int]] = json.load(f)
        else:
            self.index = {}

        # Open data file for appending new entries
        self.data_file = open(self.data_path, "a+b")
        self._mmap: Optional[mmap.mmap] = None

    def _invalidate_mmap(self) -> None:
        """Close current mmap so it gets recreated on next read."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def _get_mmap(self) -> mmap.mmap:
        """Lazily create/recreate the mmap for reading."""
        if self._mmap is None:
            self.data_file.flush()
            size = os.path.getsize(self.data_path)
            if size == 0:
                raise ValueError("Cache file is empty, nothing to read")
            # Open a separate file descriptor for read-only mmap
            fd = os.open(self.data_path, os.O_RDONLY)
            try:
                self._mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)  # mmap keeps its own reference
        return self._mmap

    def __contains__(self, word: bytes) -> bool:
        key = word.decode("utf-8") if isinstance(word, bytes) else word
        return key in self.index

    def get(self, word: bytes) -> Optional[List[TTSChunk]]:
        """
        Retrieve cached chunks for a word. Returns None on cache miss.
        """
        key = word.decode("utf-8") if isinstance(word, bytes) else word
        if key not in self.index:
            return None

        entry = self.index[key]
        offset = entry["offset"]
        length = entry["length"]

        mm = self._get_mmap()
        blob = mm[offset:offset + length]

        return self._deserialize_chunks(blob)

    def put(self, word: bytes, chunks: List[TTSChunk]) -> None:
        """
        Store chunks for a word, appending to the binary file.
        """
        key = word.decode("utf-8") if isinstance(word, bytes) else word
        if key in self.index:
            return  # already cached

        blob = self._serialize_chunks(chunks)

        self.data_file.seek(0, 2)  # seek to end
        offset = self.data_file.tell()
        self.data_file.write(blob)
        self.data_file.flush()

        self.index[key] = {"offset": offset, "length": len(blob)}
        self._invalidate_mmap()

    def _serialize_chunks(self, chunks: List[TTSChunk]) -> bytes:
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

    def _deserialize_chunks(self, blob: bytes) -> List[TTSChunk]:
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

    def save_index(self) -> None:
        """Persist the index to disk."""
        with open(self.index_path, "w") as f:
            json.dump(self.index, f)

    def close(self) -> None:
        """Flush index and close all file handles."""
        self.save_index()
        self._invalidate_mmap()
        self.data_file.close()

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "words_cached": len(self.index),
            "file_size_bytes": os.path.getsize(self.data_path)
            if os.path.exists(self.data_path) else 0,
        }

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass