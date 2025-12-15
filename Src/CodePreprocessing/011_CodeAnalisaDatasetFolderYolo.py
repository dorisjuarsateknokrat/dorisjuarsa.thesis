from pathlib import Path
from collections import Counter

# ==================================================
# PATH CONFIG (KONSISTEN DENGAN PROJECT)
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

YOLO_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "Datasets"
    / "DorisjuarsaDatasetYolo"
).resolve()

# ==================================================
# CONFIG
# ==================================================
CLASSES = ["BAS", "EOS", "LIM", "MON", "NEU"]

SPLITS = ["train", "val", "test"]
IMG_EXT = ".jpg"

# ==================================================
# ANALYSIS
# ==================================================
print("📊 ANALISIS DATASET YOLO\n")

total_images = 0
global_class_counter = Counter()

for split in SPLITS:
    img_dir = YOLO_DIR / "images" / split
    lbl_dir = YOLO_DIR / "labels" / split

    images = list(img_dir.glob(f"*{IMG_EXT}"))
    labels = list(lbl_dir.glob("*.txt"))

    assert len(images) == len(
        labels), f"❌ Mismatch image-label di split {split}"

    split_counter = Counter()

    for lbl in labels:
        # Ambil class dari PREFIX filename (BAS_xxx.txt)
        class_name = lbl.stem.split("_")[0]
        split_counter[class_name] += 1
        global_class_counter[class_name] += 1

    print(f"🔹 SPLIT: {split.upper()}")
    print(f"  Total images : {len(images)}")

    for cls in CLASSES:
        print(f"  {cls:<3}: {split_counter.get(cls, 0)}")

    print("-" * 40)

    total_images += len(images)

# ==================================================
# GLOBAL SUMMARY
# ==================================================
print("\n📌 RINGKASAN TOTAL DATASET")
print(f"TOTAL images: {total_images}")

for cls in CLASSES:
    count = global_class_counter.get(cls, 0)
    percent = (count / total_images) * 100
    print(f"{cls:<3}: {count:>5} ({percent:5.2f}%)")

print("\n✅ Analisis dataset selesai.")
