from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent

# path ke folder merge
MERGE_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "Datasets"
    / "dorisjuarsaCvatYolo1.1"
    / "data"
    / "merge"
).resolve()

# suffix yang ingin dihapus
BAD_SUFFIXES = (
    "_0.jpg", "_1.jpg", "_2.jpg", "_3.jpg", "_4.jpg",
    "_0.txt", "_1.txt", "_2.txt", "_3.txt", "_4.txt",
)

# ===== cek folder merge =====
if not MERGE_DIR.exists():
    print(f"[ERROR] Folder merge tidak ditemukan: {MERGE_DIR}")
    sys.exit(1)

# ===== DRY RUN =====
print("=== DRY RUN (tidak menghapus) ===")

files_to_delete = []

for file_path in MERGE_DIR.iterdir():
    if not file_path.is_file():
        continue

    if file_path.name.endswith(BAD_SUFFIXES):
        files_to_delete.append(file_path)
        print("AKAN DIHAPUS:", file_path.name)

print(f"\nTotal file terdeteksi: {len(files_to_delete)}")

# ===== konfirmasi =====
confirm = input("\nKetik 'YES' untuk benar-benar menghapus: ")

if confirm != "YES":
    print("Dibatalkan.")
    sys.exit(0)

# ===== HAPUS FILE =====
deleted = 0
for file_path in files_to_delete:
    file_path.unlink()
    deleted += 1

print("\n=== SELESAI ===")
print(f"File terhapus: {deleted}")
