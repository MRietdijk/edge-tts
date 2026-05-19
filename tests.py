from pathlib import Path
from random import sample
import subprocess
from typing import List

INPUT_DIR = Path("./test_data")
MEDIA_OUTPUT = Path("./media.mp3")
OUTPUT_DIR = Path("./results")
SAMPLE_SIZE = 1
ENERGY_OUT_FILE = "energy.txt"
CACHE_OUT_FILE = "cache_stats.txt"


def getSampleSet() -> List[Path]:
    return [p for p in INPUT_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]

def getSample() -> List[Path]:
    return sample(getSampleSet(), k=SAMPLE_SIZE)


def runFile(index, path: Path) -> None:
    energy_file = f"{index}_{ENERGY_OUT_FILE}"
    cache_stats_file = f"{index}_{CACHE_OUT_FILE}"
    process = subprocess.Popen(
        # ["perf", "stat", "-e", "power/energy-pkg/,power/energy-cores/", "-o", OUTPUT_DIR.joinpath(energy_file).resolve(), "edge-tts", "--enable-caching", "--file", path.resolve(), "--write-media", MEDIA_OUTPUT.resolve()],
        ["edge-tts", "--enable-caching", "--file", path.resolve(), "--write-media", MEDIA_OUTPUT.resolve()],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = process.communicate()
    cache_stats_file_path = OUTPUT_DIR.joinpath(cache_stats_file)
    with open(cache_stats_file_path, "w", encoding="utf-8") as file:
        file.write(f"STAT: {path}\t{path.stat().st_size}\n\nERR: {stderr}\n\nOUT: {stdout}")

def run():
    for i, file in enumerate(getSample()):
        print(file, file.stat().st_size)
        runFile(i, file)

run()