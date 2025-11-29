from PIL import Image  # untuk cek ukuran gambar
import matplotlib.pyplot as plt
from collections import Counter
import yaml
import glob
import os

# Root dataset (ubah jika perlu)
DATASET_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/DorisjuarsaDatasetBalance360x360_rotate"

# Label dirs langsung ditentukan
label_dirs = {
    "train": os.path.join(DATASET_ROOT, "labels/train"),
    "val": os.path.join(DATASET_ROOT, "labels/val"),
    "test": os.path.join(DATASET_ROOT, "labels/test"),
}

# Image dirs langsung ditentukan
image_dirs = {
    "train": os.path.join(DATASET_ROOT, "images/train"),
    "val": os.path.join(DATASET_ROOT, "images/val"),
    "test": os.path.join(DATASET_ROOT, "images/test"),
}

# Load class names dari data.yaml (hanya untuk mapping index → nama)
with open(os.path.join(DATASET_ROOT, "data.yaml"), "r") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml["names"]

# Counter distribusi kelas global dan per split
global_counts = Counter()
split_counts = {split: Counter() for split in label_dirs}

# Counter ukuran gambar global dan per split
global_sizes = Counter()
split_sizes = {split: Counter() for split in image_dirs}

# Untuk cek koordinat
min_coords = {"x": 1.0, "y": 1.0, "w": 1.0, "h": 1.0}
invalid_labels = []
mismatch_labels = []

# Loop semua file label
for split, d in label_dirs.items():
    for file in glob.glob(os.path.join(d, "*.txt")):
        filename = os.path.basename(file)
        expected_class_str = filename[:3]  # ambil 3 digit pertama nama file
        try:
            expected_class_id = int(expected_class_str)
        except ValueError:
            expected_class_id = None  # kalau tidak bisa dikonversi ke angka

        with open(file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = parts
                # cls = int(cls)
                cls = int(float(cls))
                x, y, w, h = map(float, [x, y, w, h])

                # Hitung jumlah class
                global_counts[cls] += 1
                split_counts[split][cls] += 1

                # Update koordinat minimum
                min_coords["x"] = min(min_coords["x"], x)
                min_coords["y"] = min(min_coords["y"], y)
                min_coords["w"] = min(min_coords["w"], w)
                min_coords["h"] = min(min_coords["h"], h)

                # Cek invalid (koordinat harus 0–1)
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    invalid_labels.append((file, line.strip()))

                # Cek apakah class sesuai dengan nama file
                if expected_class_id is not None and cls != expected_class_id:
                    mismatch_labels.append(
                        (file, line.strip(), expected_class_id, cls))

# Loop semua file gambar
for split, d in image_dirs.items():
    for file in glob.glob(os.path.join(d, "*.*")):
        try:
            with Image.open(file) as img:
                size = img.size  # (width, height)
                global_sizes[size] += 1
                split_sizes[split][size] += 1
        except Exception as e:
            print(f"Gagal membaca {file}: {e}")

# Print distribusi global
print("Distribusi Class (Global):")
total_global = 0
for cls_id, count in sorted(global_counts.items()):
    print(f"Class {cls_id} ({class_names[cls_id]}): {count}")
    total_global += count
print(f"Total (Global): {total_global}")

# Print distribusi per split
for split, counts in split_counts.items():
    print(f"\nDistribusi Class ({split}):")
    total_split = 0
    for cls_id, count in sorted(counts.items()):
        print(f"Class {cls_id} ({class_names[cls_id]}): {count}")
        total_split += count
    print(f"Total ({split}): {total_split}")

# Print ukuran gambar global
print("\nDistribusi ukuran gambar (Global):")
for size, count in global_sizes.most_common():
    print(f"{size[0]}x{size[1]}: {count} file")
print(f"Total gambar: {sum(global_sizes.values())}")

# Print ukuran gambar per split
for split, sizes in split_sizes.items():
    print(f"\nDistribusi ukuran gambar ({split}):")
    for size, count in sizes.most_common():
        print(f"{size[0]}x{size[1]}: {count} file")
    print(f"Total ({split}): {sum(sizes.values())}")

# Print koordinat minimum
print("\nKoordinat minimum yang valid ditemukan:")
print(min_coords)

# Print label invalid
if invalid_labels:
    print("\nLabel invalid ditemukan:")
    for f, line in invalid_labels:
        print(f"{f}: {line}")
else:
    print("\nSemua label valid.")

# Print mismatch class vs nama file
if mismatch_labels:
    print("\nMismatch Class vs Filename ditemukan:")
    for f, line, expected, found in mismatch_labels:
        print(f"{f}: {line} | Expected: {expected}, Found: {found}")
else:
    print("\nSemua label sesuai dengan nama file.")

# Plot distribusi global
plt.bar([class_names[i] for i in global_counts.keys()], global_counts.values())
plt.xlabel("Class")
plt.ylabel("Jumlah")
plt.title("Distribusi Class (Global)")
plt.show()
