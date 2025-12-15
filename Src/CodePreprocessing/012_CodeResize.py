from pathlib import Path
import random
import shutil
from PIL import Image

# ==================================================
# PATH CONFIG
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "Datasets"
    / "dorisjuarsaCvatYolo1.1"
    / "data"
).resolve()

# SOURCE DATA (BASE IMAGE 360x360)
DATASET_DIR = DATA_DIR / "merge_resize_360x360"

# OUTPUT DATASET (RESIZED)
NEW_SIZE = 128
YOLO_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "Datasets"
    / f"DorisjuarsaDatasetYoloBaseSize_{NEW_SIZE}"
).resolve()

YAML_PATH = YOLO_DIR / "data.yaml"

# ==================================================
# CONFIG
# ==================================================
CLASSES = ["BAS", "EOS", "LIM", "MON", "NEU"]
RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
IMG_EXT = ".jpg"

random.seed(42)

# ==================================================
# CREATE YOLO STRUCTURE
# ==================================================
for split in RATIOS:
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# ==================================================
# SPLIT + RESIZE PROCESS
# ==================================================
print("🔄 Mulai split + resize dataset...\n")

for cls in CLASSES:
    images = sorted(DATASET_DIR.glob(f"{cls}_*{IMG_EXT}"))
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * RATIOS["train"])
    n_val = int(n_total * RATIOS["val"])

    split_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    print(f"Class {cls} | Total: {n_total}")

    for split, files in split_map.items():
        print(f"  {split:<5}: {len(files)}")

        for img_path in files:
            lbl_path = img_path.with_suffix(".txt")
            if not lbl_path.exists():
                raise FileNotFoundError(f"Label missing: {lbl_path.name}")

            # === RESIZE IMAGE ===
            with Image.open(img_path) as img:
                resized = img.resize((NEW_SIZE, NEW_SIZE), Image.LANCZOS)
                resized.save(
                    YOLO_DIR / "images" / split / img_path.name
                )

            # === COPY LABEL (YOLO normalized → TIDAK DIUBAH) ===
            shutil.copy2(
                lbl_path,
                YOLO_DIR / "labels" / split / lbl_path.name
            )

# ==================================================
# SANITY CHECK
# ==================================================
print("\n🔍 Sanity check:")

total_img = total_lbl = 0
for split in ["train", "val", "test"]:
    n_img = len(list((YOLO_DIR / "images" / split).glob("*.jpg")))
    n_lbl = len(list((YOLO_DIR / "labels" / split).glob("*.txt")))
    print(f"{split:<5}: images={n_img}, labels={n_lbl}")
    total_img += n_img
    total_lbl += n_lbl

assert total_img == total_lbl, "❌ Image–label mismatch!"

print(f"\n📊 TOTAL images={total_img}, labels={total_lbl}")

# ==================================================
# AUTO GENERATE data.yaml
# ==================================================
yaml_text = f"""# YOLOv8 dataset configuration
path: {YOLO_DIR.as_posix()}

train: images/train
val: images/val
test: images/test

names:
"""
for i, cls in enumerate(CLASSES):
    yaml_text += f"  {i}: {cls}\n"

YAML_PATH.write_text(yaml_text)

print(f"\n📝 data.yaml dibuat di:\n{YAML_PATH}")
print("\n✅ DONE: Split + resize dataset siap untuk training YOLO.")
