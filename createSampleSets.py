from pathlib import Path
from random import shuffle
from typing import List

N_SETS = 20
SAMPE_SIZE = 100000 # 100kb in bytes.

INPUT_DIR = Path("./test_data")
SAMPLES_DIR = Path('./samples')

def getInputSet() -> List[Path]:
    return [p for p in INPUT_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]

def getSample() -> List[Path]:
    perm = getInputSet()
    shuffle(perm)

    size = 0
    sample = []

    index = 0
    while size < SAMPE_SIZE:
        file = perm[index]
        sample.append(file)
        size += file.stat().st_size
        index += 1

    return sample

def createSamples():
    SAMPLES_DIR.mkdir(exist_ok=True)
    i = 0
    while i < N_SETS:
        path = SAMPLES_DIR.joinpath(Path(f"sample_{i}.txt"))
        data = [str(p.relative_to('./')) for p in getSample()]
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(data))

        i += 1

createSamples()