import kagglehub
import shutil
import os

path = kagglehub.dataset_download("hari31416/food-101")

FOLDERS_TO_KEEP = [
    "beef_tartare",
    "chicken_quesadilla",
    "risotto",
    "spaghetti_carbonara",
    "pancakes"
]

OUTPUT_DIR = "./food_subset"

for split in ["train", "validation"]:
    for folder in FOLDERS_TO_KEEP:
        src = os.path.join(path, "food-101", split, folder)
        dst = os.path.join(OUTPUT_DIR, split, folder)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Copied: {split}/{folder}")
        else:
            print(f"Not found: {split}/{folder}")