from pathlib import Path
import subprocess
from typing import List

SAMPLE_DIR = Path("./samples")
MEDIA_OUTPUT = Path("./media.mp3")
OUTPUT_DIR = Path("./results")

ENERGY_OUT_FILE = "energy.txt"
CACHE_OUT_FILE = "cache_stats.txt"



def runFile(index, path: Path, results_dir: Path) -> None:
    energy_file = f"{index}_{ENERGY_OUT_FILE}"
    cache_stats_file = f"{index}_{CACHE_OUT_FILE}"
    # process = subprocess.Popen(
    #     # ["perf", "stat", "-e", "power/energy-pkg/,power/energy-cores/", "-o", results_dir.joinpath(energy_file).resolve(), "edge-tts", "--enable-caching", "--file", path.resolve(), "--write-media", MEDIA_OUTPUT.resolve()],
    #     ["edge-tts", "--enable-caching", "--file", path.resolve(), "--write-media", MEDIA_OUTPUT.resolve()],
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True
    # )

    # stdout, stderr = process.communicate()
    stdout, stderr = ("aaa", "bbb")
    cache_stats_file_path = results_dir.joinpath(cache_stats_file)
    with open(cache_stats_file_path, "w", encoding="utf-8") as file:
        file.write(f"STAT: {path}\t{path.stat().st_size}\n\nERR: {stderr}\n\nOUT: {stdout}")

def run():
    for i, p in enumerate(SAMPLE_DIR.iterdir()):
        paths = getPathsFromFile(p)
        results_dir = OUTPUT_DIR.joinpath(Path(f"results_{i}/"))
        results_dir.mkdir(exist_ok=True, parents=True)
        for j, input in enumerate(paths):
            runFile(j, input, results_dir)

def getPathsFromFile(filePath: Path) -> List[Path]:
    with open (filePath, 'r', encoding='utf-8') as f:
        data = f.read()
        return [Path(line) for line in data.splitlines()]

run()