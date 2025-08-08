import os
from pathlib import Path


ROOT_PATH = Path('/content/drive/MyDrive/datasets/sen2venus')

TACO_RAW_DIR = ROOT_PATH / 'TACO_raw_data'
os.makedirs(TACO_RAW_DIR, exist_ok=True)
print(f"Data will be saved to: {TACO_RAW_DIR}")

SELECTED_SUBSETS = [
    "SUDOUE-4",
    "SUDOUE-5",
    "SUDOUE-6"
]
TACO_FILE_PATHS = [TACO_RAW_DIR / f"{site_name}.taco" for site_name in SELECTED_SUBSETS]


NORMALIZED_SETS_DIR = ROOT_PATH / 'normalized_sets'
os.makedirs(NORMALIZED_SETS_DIR, exist_ok=True)
print(f"Normalaized datest will be saved to: {NORMALIZED_SETS_DIR}")

TRAIN_SAVE_DIR = NORMALIZED_SETS_DIR / 'train'
os.makedirs(TRAIN_SAVE_DIR, exist_ok=True)
print(f"Train data will be saved to: {TRAIN_SAVE_DIR}")

VAL_SAVE_DIR = NORMALIZED_SETS_DIR / 'val'
os.makedirs(VAL_SAVE_DIR, exist_ok=True)
print(f"Validation data will be saved to: {VAL_SAVE_DIR}")

TEST_SAVE_DIR = NORMALIZED_SETS_DIR / 'test'
os.makedirs(TEST_SAVE_DIR, exist_ok=True)
print(f"Test data will be saved to: {TEST_SAVE_DIR}")