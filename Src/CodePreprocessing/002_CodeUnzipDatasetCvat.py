

from pathlib import Path
import zipfile
import sys

BASE_DIR = Path(__file__).resolve().parent

zip_path = (BASE_DIR / ".." / ".." / "Data" / "Datasets" /
            "dorisjuarsaCvatYolo1.1" / "dorisjuarsaCvatYolo1.1.zip").resolve()
extract_to = (BASE_DIR / ".." / ".." / "Data" / "Datasets" /
              "dorisjuarsaCvatYolo1.1" / "data").resolve()

# cek apakah file zip ada
if not zip_path.exists():
    print(f"[ERROR] File ZIP tidak ditemukan: {zip_path}")
    sys.exit(1)

# cek apakah benar-benar file zip
if not zipfile.is_zipfile(zip_path):
    print(f"[ERROR] File ditemukan tapi bukan ZIP: {zip_path}")
    sys.exit(1)

# buat folder output jika belum ada
extract_to.mkdir(parents=True, exist_ok=True)

# proses unzip
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[OK] Unzip selesai ke: {extract_to}")

except Exception as e:
    print(f"[ERROR] Gagal unzip: {e}")
    sys.exit(1)
