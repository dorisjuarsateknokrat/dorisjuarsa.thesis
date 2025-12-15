from pathlib import Path
import random
import shutil

# ==================================================
# PATH CONFIG (KONSISTEN DENGAN PROJECT)
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

# SOURCE: base images + labels (SATU FOLDER)
DATASET_DIR = DATA_DIR / "merge_resize_360x360"

# OUTPUT: dataset YOLO (FORMAT images/train, labels/train)
YOLO_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "Datasets"
    / "DorisjuarsaDatasetYolo"
).resolve()

# OUTPUT YAML (AUTO GENERATED)
YAML_PATH = YOLO_DIR / "data.yaml"

# ==================================================
# CONFIG
# ==================================================
CLASSES = ["BAS", "EOS", "LIM", "MON", "NEU"]

RATIOS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

IMG_EXT = ".jpg"
random.seed(42)

# ==================================================
# CREATE YOLO DIRECTORY STRUCTURE
# ==================================================
for split in RATIOS.keys():
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# ==================================================
# SPLITTING PROCESS (STRATIFIED BY PREFIX)
# ==================================================
print("🔄 Mulai proses splitting dataset...\n")

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
                raise FileNotFoundError(
                    f"❌ Label tidak ditemukan untuk image: {img_path.name}"
                )

            shutil.copy2(
                img_path,
                YOLO_DIR / "images" / split / img_path.name
            )
            shutil.copy2(
                lbl_path,
                YOLO_DIR / "labels" / split / lbl_path.name
            )

# ==================================================
# SANITY CHECK
# ==================================================
print("\n🔍 Sanity check jumlah file:")

total_images = 0
total_labels = 0

for split in ["train", "val", "test"]:
    img_count = len(list((YOLO_DIR / "images" / split).glob("*.jpg")))
    lbl_count = len(list((YOLO_DIR / "labels" / split).glob("*.txt")))

    total_images += img_count
    total_labels += lbl_count

    print(f"{split:<5}: images = {img_count}, labels = {lbl_count}")

assert total_images == total_labels, "❌ Jumlah image dan label tidak sama!"

print(f"\n📊 TOTAL images = {total_images}, labels = {total_labels}")

# ==================================================
# AUTO GENERATE data.yaml
# ==================================================
yaml_content = f"""# YOLOv8 dataset configuration
path: {YOLO_DIR.as_posix()}

train: images/train
val: images/val
test: images/test

names:
"""

for idx, cls in enumerate(CLASSES):
    yaml_content += f"  {idx}: {cls}\n"

YAML_PATH.write_text(yaml_content)

print(f"\n📝 data.yaml berhasil dibuat di:\n{YAML_PATH}")
print("\n✅ DONE: Dataset & data.yaml siap untuk training YOLO.")
