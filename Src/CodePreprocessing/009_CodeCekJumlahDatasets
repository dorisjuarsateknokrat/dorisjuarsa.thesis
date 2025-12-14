from pathlib import Path
import sys
from collections import Counter

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

# ganti ke dataset final kamu
DATASET_DIR = DATA_DIR / "merge_resize_360x360"

if not DATASET_DIR.exists():
    print(f"[ERROR] Folder dataset tidak ditemukan: {DATASET_DIR}")
    sys.exit(1)

class_counter = Counter()
missing_pair = 0
invalid_name = 0

for img_path in DATASET_DIR.glob("*.jpg"):
    txt_path = DATASET_DIR / f"{img_path.stem}.txt"

    if not txt_path.exists():
        missing_pair += 1
        continue

    # format harus ada "_"
    if "_" not in img_path.stem:
        invalid_name += 1
        continue

    # prefix kelas = sebelum "_"
    cls = img_path.stem.split("_", 1)[0]
    class_counter[cls] += 1

# ===== OUTPUT =====
print("\n=== JUMLAH DATA PER KELAS ===")
total = 0
for cls, count in sorted(class_counter.items()):
    print(f"{cls:>4} : {count}")
    total += count

print("-----------------------------")
print(f"TOTAL : {total}")

if missing_pair:
    print(f"\n[WARNING] JPG tanpa pasangan TXT : {missing_pair}")

if invalid_name:
    print(
        f"[WARNING] Nama file tidak sesuai format <KELAS>_* : {invalid_name}")
