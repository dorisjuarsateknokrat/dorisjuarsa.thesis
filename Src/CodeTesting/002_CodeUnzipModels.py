#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
001_CodeUnzipModels.py

- Mengekstrak file ZIP model ke folder Data/DataModels/runs
- Menghindari folder dobel (ZIP sudah punya root folder)
- Mendukung lebih dari satu file ZIP
- Tidak mengizinkan overwrite folder hasil ekstraksi
"""

from pathlib import Path
import zipfile
import sys

# ==================================================
# BASE PATH
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# MODELS PATH
# ==================================================
MODELS_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "DataModels"
    / "runs"
).resolve()

# ==================================================
# MODEL ZIP FILES
# ==================================================
MODEL_ZIPS = [
    "DorisjuarsaDatasetYoloBaseSizeToScale0_25_640_small.zip",
    "DorisjuarsaDatasetYoloBaseSizeToScale0_25Clahe_640_small.zip",
]

# ==================================================
# VALIDATION: MODELS DIR
# ==================================================
if not MODELS_DIR.exists():
    print("❌ Folder runs tidak ditemukan.")
    print("👉 Jalankan 001_DownloadModels.py terlebih dahulu.")
    sys.exit(1)

# ==================================================
# UNZIP PROCESS (ANTI DOBEL FOLDER)
# ==================================================
for zip_name in MODEL_ZIPS:
    zip_path = MODELS_DIR / zip_name

    if not zip_path.exists():
        print(f"❌ File ZIP tidak ditemukan: {zip_path}")
        sys.exit(1)

    print("\n📦 Mengekstrak model:")
    print(f"SOURCE: {zip_path}")
    print(f"DEST  : {MODELS_DIR}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Ambil semua path di dalam ZIP
            members = zip_ref.namelist()

            # Deteksi root folder ZIP
            root_folders = {
                m.split("/")[0]
                for m in members
                if "/" in m
            }

            if len(root_folders) != 1:
                print("❌ Struktur ZIP tidak valid (harus 1 root folder).")
                print(f"   Root terdeteksi: {root_folders}")
                sys.exit(1)

            root_folder = next(iter(root_folders))
            target_dir = MODELS_DIR / root_folder

            if target_dir.exists():
                print(f"⚠️ Folder hasil ekstraksi sudah ada:")
                print(f"{target_dir}")
                print("❌ Overwrite tidak diizinkan.")
                sys.exit(1)

            zip_ref.extractall(MODELS_DIR)

        print("✅ Ekstraksi selesai.")
        print(f"📁 Output: {target_dir}")

    except zipfile.BadZipFile:
        print("❌ File ZIP rusak atau tidak valid:")
        print(zip_path)
        sys.exit(1)

print("\n🎉 Semua model berhasil diekstrak dengan struktur rapi.")
