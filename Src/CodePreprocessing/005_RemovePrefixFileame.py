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

# cek folder merge
if not MERGE_DIR.exists():
    print(f"[ERROR] Folder merge tidak ditemukan: {MERGE_DIR}")
    sys.exit(1)

renamed = 0
skipped = 0

for file_path in MERGE_DIR.iterdir():
    if not file_path.is_file():
        continue

    stem = file_path.stem      # nama tanpa ekstensi
    suffix = file_path.suffix  # .jpg / .txt

    # harus ada underscore
    if "_" not in stem:
        continue

    prefix, rest = stem.split("_", 1)

    # cek apakah prefix numerik
    if not prefix.isdigit():
        continue

    new_name = rest + suffix
    new_path = MERGE_DIR / new_name

    # skip jika nama tujuan sudah ada
    if new_path.exists():
        skipped += 1
        continue

    file_path.rename(new_path)
    renamed += 1

print("\n=== SELESAI ===")
print(f"File di-rename : {renamed}")
print(f"File di-skip   : {skipped}")
