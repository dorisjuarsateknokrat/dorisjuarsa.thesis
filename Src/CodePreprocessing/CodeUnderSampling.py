
import os
import glob
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Root dataset asli
DATASET_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/datasetDorisjuarsaBalance"

# Tanya persentase undersampling
try:
    percent = int(
        input("Masukkan persentase data yang ingin dipakai (contoh: 50 untuk 50%): "))
    if not (1 <= percent <= 100):
        raise ValueError
except ValueError:
    print("❌ Input tidak valid. Masukkan angka 1–100.")
    exit(1)

# Hitung nama folder output
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_ROOT = f"{DATASET_ROOT}_Undersampling_{percent}pct_{timestamp}"

# Buat struktur folder baru (images & labels untuk train/val/test)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

# Copy data.yaml ke folder baru
yaml_src = os.path.join(DATASET_ROOT, "data.yaml")
yaml_dst = os.path.join(OUTPUT_ROOT, "data.yaml")
if os.path.exists(yaml_src):
    shutil.copy(yaml_src, yaml_dst)


def copy_file(src, dst):
    shutil.copy(src, dst)


def process_split(split):
    img_dir = os.path.join(DATASET_ROOT, "images", split)
    label_dir = os.path.join(DATASET_ROOT, "labels", split)

    out_img_dir = os.path.join(OUTPUT_ROOT, "images", split)
    out_label_dir = os.path.join(OUTPUT_ROOT, "labels", split)

    # Kelompokkan file berdasarkan class id dari nama file (3 digit pertama)
    class_files = {}
    for img_path in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
        filename = os.path.basename(img_path)
        class_id = filename[:3]
        class_files.setdefault(class_id, []).append(filename)

    tasks = []
    with ProcessPoolExecutor() as executor:
        for class_id, files in class_files.items():
            total_files = len(files)
            keep_count = max(1, int(total_files * percent / 100))

            selected_files = files[:keep_count]  # incremental, bukan random
            for filename in selected_files:
                img_path = os.path.join(img_dir, filename)
                label_path = os.path.join(
                    label_dir, os.path.splitext(filename)[0] + ".txt")

                out_img_path = os.path.join(out_img_dir, filename)
                out_label_path = os.path.join(
                    out_label_dir, os.path.splitext(filename)[0] + ".txt")

                tasks.append(executor.submit(
                    copy_file, img_path, out_img_path))
                if os.path.exists(label_path):
                    tasks.append(executor.submit(
                        copy_file, label_path, out_label_path))

    print(
        f"Split {split}: {len(tasks)//2} file dipilih ({percent}% dari tiap class).")


# Proses semua split
for split in ["train", "val", "test"]:
    process_split(split)

print("✅ Undersampling selesai.")
print(f"Hasil tersimpan di: {OUTPUT_ROOT}")
