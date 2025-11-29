from glob import glob
import shutil
import os

# Path dataset asal dan tujuan
SRC_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/datasetDorisjuarsaBalance"
DST_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/datasetDorisjuarsaBalanceMergeRaw"

# Buat folder target
images_merge = os.path.join(DST_ROOT, "ImagesMerge")
labels_merge = os.path.join(DST_ROOT, "LabelsMerge")
os.makedirs(images_merge, exist_ok=True)
os.makedirs(labels_merge, exist_ok=True)

# Copy semua images
for split in ["train", "val", "test"]:
    files = glob(os.path.join(SRC_ROOT, "images", split, "*.*"))
    for f in files:
        dst = os.path.join(images_merge, os.path.basename(f))
        shutil.copy(f, dst)

# Copy semua labels
for split in ["train", "val", "test"]:
    files = glob(os.path.join(SRC_ROOT, "labels", split, "*.txt"))
    for f in files:
        dst = os.path.join(labels_merge, os.path.basename(f))
        shutil.copy(f, dst)

print("✅ Semua file sudah digabung ke folder:")
print("Images:", images_merge)
print("Labels:", labels_merge)
