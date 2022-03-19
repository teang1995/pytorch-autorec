from pathlib import Path
from os import path

# Directories
ROOT_DIR = Path(__file__).resolve().parent.__str__()
DATA_DIR = path.join(ROOT_DIR, "data")

# checkpoint dir
CHECKPOINT_DIR = path.join(ROOT_DIR, "checkpoints")

# logging dir
LOGGING_DIR = path.join(ROOT_DIR, "logging")
if __name__ == "__main__":
    print(ROOT_DIR)
    print(DATA_DIR)
    print(CHECKPOINT_DIR)