import os
import shutil
from glob import glob
import random
import yaml

# Path folder merge awal
SRC_IMAGES = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/datasetDorisjuarsaBalanceMergeRaw/ImagesMerge"
SRC_LABELS = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/datasetDorisjuarsaBalanceMergeRaw/LabelsMerge"

# Path dataset baru
DST_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/datasetDorisjuarsaBalance_NEU_1200"
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DST_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DST_ROOT, "labels", split), exist_ok=True)

# Kelas lain tetap
other_classes = ["BAS", "EOS", "LIM", "MON"]

# ======================
# 1️⃣ Proses NEB/NES → pilih subset
# ======================
# Ambil semua NEB/NES
neb_files = sorted(glob(os.path.join(SRC_IMAGES, "NEB_*.jpg")))
nes_files = sorted(glob(os.path.join(SRC_IMAGES, "NES_*.jpg")))

# Pilih subset sesuai rencana
neb_keep = neb_files[:600]       # 0001–0600
nes_keep = nes_files[600:]       # 0601–1200

# Gabungkan → shuffle
neu_files = neb_keep + nes_keep
random.seed(42)
random.shuffle(neu_files)

# Buat daftar pasangan label + image
neu_pairs = []
for img_path in neu_files:
    base = os.path.basename(img_path).replace(".jpg", "")
    label_path = os.path.join(SRC_LABELS, base + ".txt")
    neu_pairs.append((label_path, img_path))

# ======================
# 2️⃣ Reindex & split 70/20/10
# ======================
total_neu = len(neu_pairs)
train_end = int(0.7 * total_neu)
val_end = int(0.9 * total_neu)

for idx, (lbl, img) in enumerate(neu_pairs, start=1):
    new_base = f"NEU_{idx:04d}"
    # Tentukan split
    if idx <= train_end:
        split = "train"
    elif idx <= val_end:
        split = "val"
    else:
        split = "test"
    # Copy image
    dst_img = os.path.join(DST_ROOT, "images", split, new_base + ".jpg")
    shutil.copy(img, dst_img)
    # Update label class jadi 2 (NEU)
    with open(lbl, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        new_lines.append(f"2 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
    dst_lbl = os.path.join(DST_ROOT, "labels", split, new_base + ".txt")
    with open(dst_lbl, "w") as f:
        f.writelines(new_lines)

# ======================
# 3️⃣ Proses kelas lain → copy & mapping class tetap
# ======================
class_map = {"BAS": 0, "EOS": 1, "LIM": 3, "MON": 4}  # NEU sudah 2

for cls in other_classes:
    cls_files = sorted(glob(os.path.join(SRC_IMAGES, f"{cls}_*.jpg")))
    for img_path in cls_files:
        base = os.path.basename(img_path).replace(".jpg", "")
        label_path = os.path.join(SRC_LABELS, base + ".txt")
        # Tentukan split dari urutan
        num = int(base.split("_")[-1])
        if num <= 840:
            split = "train"
        elif num <= 1080:
            split = "val"
        else:
            split = "test"
        # Copy image
        dst_img = os.path.join(DST_ROOT, "images", split, base + ".jpg")
        shutil.copy(img_path, dst_img)
        # Update label
        with open(label_path, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = class_map[cls]
            new_lines.append(
                f"{cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
        dst_lbl = os.path.join(DST_ROOT, "labels", split, base + ".txt")
        with open(dst_lbl, "w") as f:
            f.writelines(new_lines)

# ======================
# 4️⃣ Buat data.yaml baru
# ======================
data_yaml = {
    "path": DST_ROOT,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "BAS", 1: "EOS", 2: "NEU", 3: "LIM", 4: "MON"}
}

with open(os.path.join(DST_ROOT, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f)

print("✅ Dataset NEU_1200 berhasil dibuat di:", DST_ROOT)
