from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent

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

if not MERGE_DIR.exists():
    print(f"[ERROR] Folder merge tidak ditemukan: {MERGE_DIR}")
    sys.exit(1)

# ===== mapping kelas =====
CLASS_MAP = {
    "BA": "BAS",
    "EO": "EOS",
    "BNE": "NEU",
    "SNE": "NEU",
    "NEUTROPHIL": "NEU",
    "LY": "LIM",
    "MO": "MON",
}

# ===== kumpulkan pasangan YOLO =====
pairs_by_class = {}
missing_txt = []

for jpg_path in MERGE_DIR.glob("*.jpg"):
    stem = jpg_path.stem
    txt_path = MERGE_DIR / f"{stem}.txt"

    if not txt_path.exists():
        missing_txt.append(jpg_path.name)
        continue

    if "_" not in stem:
        continue

    prefix, _ = stem.split("_", 1)

    if prefix not in CLASS_MAP:
        continue

    new_class = CLASS_MAP[prefix]
    pairs_by_class.setdefault(new_class, []).append((jpg_path, txt_path))

# ===== laporkan pasangan bermasalah =====
if missing_txt:
    print("\n[WARNING] JPG tanpa pasangan TXT:")
    for f in missing_txt:
        print(" -", f)

# ===== rename dua tahap =====
print("\n=== PROSES RENAME ===")

temp_pairs = []

for cls, pairs in pairs_by_class.items():
    pairs.sort(key=lambda x: x[0].name)

    for idx, (jpg, txt) in enumerate(pairs, start=1):
        new_base = f"{cls}_{idx:04d}"

        tmp_jpg = MERGE_DIR / f"__tmp__{new_base}.jpg"
        tmp_txt = MERGE_DIR / f"__tmp__{new_base}.txt"

        jpg.rename(tmp_jpg)
        txt.rename(tmp_txt)

        temp_pairs.append((tmp_jpg, tmp_txt, new_base))

# ===== rename final =====
renamed = 0
skipped = 0

for tmp_jpg, tmp_txt, new_base in temp_pairs:
    final_jpg = MERGE_DIR / f"{new_base}.jpg"
    final_txt = MERGE_DIR / f"{new_base}.txt"

    if final_jpg.exists() or final_txt.exists():
        skipped += 1
        tmp_jpg.rename(MERGE_DIR / tmp_jpg.name.replace("__tmp__", ""))
        tmp_txt.rename(MERGE_DIR / tmp_txt.name.replace("__tmp__", ""))
        continue

    tmp_jpg.rename(final_jpg)
    tmp_txt.rename(final_txt)
    renamed += 1

# ===== summary =====
print("\n=== SELESAI ===")
print(f"Pasangan di-rename : {renamed}")
print(f"Pasangan di-skip   : {skipped}")
print(f"Kelas terproses    : {list(pairs_by_class.keys())}")
