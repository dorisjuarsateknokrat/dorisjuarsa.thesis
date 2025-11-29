from tqdm import tqdm
from glob import glob
import yaml
import random
import shutil
import os

# Path dataset asal dan tujuan
SRC_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/datasetDorisjuarsaBalance"
DST_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/datasetDorisjuarsaBalance_MergedNeu"

# Buat struktur folder YOLO di dataset baru
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DST_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DST_ROOT, "labels", split), exist_ok=True)

# Mapping class lama -> baru
# 0:BAS, 1:EOS, 2:NEUB, 3:NEUS, 4:LIM, 5:MON
# NEUB & NEUS digabung ke 2:NEU
class_map = {
    0: 0,  # BAS
    1: 1,  # EOS
    2: 2,  # NEUB -> NEU
    3: 2,  # NEUS -> NEU
    4: 3,  # LIM
    5: 4   # MON
}

# Kelas nama baru
new_class_names = {
    0: "BAS",
    1: "EOS",
    2: "NEU",
    3: "LIM",
    4: "MON"
}

# Ambil semua file label
all_labels = []
for split in ["train", "val", "test"]:
    label_files = glob(os.path.join(SRC_ROOT, "labels", split, "*.txt"))
    for lf in label_files:
        img_file = os.path.join(SRC_ROOT, "images", split,
                                os.path.basename(lf).replace(".txt", ".jpg"))
        all_labels.append((lf, img_file))

# Pisahkan NEB/NES dan kelas lain
neu_files = []
other_files = []
for lf, img in all_labels:
    if "NEB" in os.path.basename(lf) or "NES" in os.path.basename(lf):
        neu_files.append((lf, img))
    else:
        other_files.append((lf, img))

print(f"Jumlah NEB+NES (NEU): {len(neu_files)}")
print(f"Jumlah kelas lain   : {len(other_files)}")

# Shuffle NEU supaya seimbang
random.seed(42)
random.shuffle(neu_files)

# Reindex NEU (0001 - 2400)
renamed_neu = []
for idx, (lf, img) in enumerate(neu_files, start=1):
    new_base = f"NEU_{idx:04d}"
    renamed_neu.append((lf, img, new_base))

# Helper fungsi untuk copy dan ubah label


def process_and_copy(lf, img, dst_split, new_base):
    # Baca dan update label
    with open(lf, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        cls = int(cls)
        new_cls = class_map[cls]
        new_lines.append(f"{new_cls} {x} {y} {w} {h}\n")

    # Simpan label baru
    dst_label = os.path.join(DST_ROOT, "labels", dst_split, f"{new_base}.txt")
    with open(dst_label, "w") as f:
        f.writelines(new_lines)

    # Copy gambar
    dst_img = os.path.join(DST_ROOT, "images", dst_split, f"{new_base}.jpg")
    shutil.copy(img, dst_img)


# Split NEU baru 70/20/10
total_neu = len(renamed_neu)
train_end = int(0.7 * total_neu)
val_end = int(0.9 * total_neu)

for i, (lf, img, new_base) in enumerate(tqdm(renamed_neu, desc="Processing NEU")):
    if i < train_end:
        split = "train"
    elif i < val_end:
        split = "val"
    else:
        split = "test"
    process_and_copy(lf, img, split, new_base)

# Proses kelas lain (tidak diubah namanya, hanya mapping class_id)
for lf, img in tqdm(other_files, desc="Processing other classes"):
    base = os.path.splitext(os.path.basename(lf))[0]

    # Tentukan split dari path asli
    if "/train/" in lf:
        split = "train"
    elif "/val/" in lf:
        split = "val"
    else:
        split = "test"

    # Update label
    with open(lf, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        cls = int(cls)
        new_cls = class_map[cls]
        new_lines.append(f"{new_cls} {x} {y} {w} {h}\n")

    dst_label = os.path.join(DST_ROOT, "labels", split, f"{base}.txt")
    with open(dst_label, "w") as f:
        f.writelines(new_lines)

    dst_img = os.path.join(DST_ROOT, "images", split, f"{base}.jpg")
    shutil.copy(img, dst_img)

# Buat data.yaml baru
yaml_data = {
    "path": DST_ROOT,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": new_class_names
}

with open(os.path.join(DST_ROOT, "data.yaml"), "w") as f:
    yaml.dump(yaml_data, f)

print("Dataset baru selesai dibuat di:", DST_ROOT)
