from pathlib import Path

OUTPUT_DIR = Path("./test_data")
CORPUS_DIR = Path("./corpus")

def writeFile(path: Path, content: str) -> None:
    if (path.is_file()):
        raise FileExistsError()

    with open(path, "w", encoding="utf-8") as file:
        file.write(content)

def readCorpusFile(path: Path) -> None:
    with open(path, "r", encoding="utf-8") as file:
        lines = file.read().splitlines()[1:] #skip first empty line
        for line in lines:
            [id, content] = line.split(maxsplit=1)
            writeFile(OUTPUT_DIR.joinpath(f"{id}.txt"), content)

def readCorpus() -> None:
    for path in CORPUS_DIR.iterdir():
        readCorpusFile(path)

readCorpus()
