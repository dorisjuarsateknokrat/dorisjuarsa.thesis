from pathlib import Path
import shutil
import sys

# lokasi script
BASE_DIR = Path(__file__).resolve().parent

# root data folder
DATA_DIR = (BASE_DIR / ".." / ".." / "Data" / "Datasets" /
            "dorisjuarsaCvatYolo1.1" / "data").resolve()

# folder merge
MERGE_DIR = DATA_DIR / "merge"

# daftar folder sumber
SOURCE_DIRS = [
    DATA_DIR / "obj_Train_data",
    DATA_DIR / "obj_Test_data",
    DATA_DIR / "obj_Validation_data",
]

# ===== 1. cek folder merge =====
if MERGE_DIR.exists():
    print(f"[STOP] Folder 'merge' sudah ada: {MERGE_DIR}")
    sys.exit(0)

# buat folder merge
MERGE_DIR.mkdir(parents=True)
print(f"[OK] Folder 'merge' dibuat: {MERGE_DIR}")

# ===== 2. proses pindahkan file =====
total_moved = 0
total_skipped = 0

for source_root in SOURCE_DIRS:
    if not source_root.exists():
        print(f"[WARNING] Folder tidak ditemukan: {source_root}")
        continue

    # ambil folder level 2 (folder kelas)
    for class_dir in source_root.glob("*/*"):
        if not class_dir.is_dir():
            continue

        for file_path in class_dir.iterdir():
            if not file_path.is_file():
                continue

            target_path = MERGE_DIR / file_path.name

            if target_path.exists():
                total_skipped += 1
                continue

            shutil.move(str(file_path), str(target_path))
            total_moved += 1


print("\n=== SELESAI ===")
print(f"File dipindahkan : {total_moved}")
print(f"File di-skip     : {total_skipped}")
print(f"Lokasi merge     : {MERGE_DIR}")
