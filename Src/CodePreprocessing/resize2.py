import os
import glob
import random
import shutil
from datetime import datetime
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor

# === KONFIGURASI ===
ORIGINAL_DATASET_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/DorisjuarsaDatasetBalance360x360Rotate"

# Parameter resize
NEW_SIZE = 128  # Ukuran baru: 128x128
OLD_SIZE = 360  # Ukuran asli: 360x360

# Buat folder output baru
timestamp = datetime.now().strftime("%y%m%d%H%M")
NEW_DATASET_ROOT = f"{ORIGINAL_DATASET_ROOT}_Resized{NEW_SIZE}_{timestamp}"

print(f"📁 Folder dataset hasil resize: {NEW_DATASET_ROOT}")

# Fungsi transformasi


def process_file(img_path, label_path, out_img_path, out_label_path, out_bb_path, new_size=128, old_size=360):
    # Proses gambar: resize SELURUH gambar ke ukuran baru
    with Image.open(img_path) as img:
        # Pastikan gambar asli berukuran old_size x old_size
        if img.size != (old_size, old_size):
            img = img.resize((old_size, old_size), Image.LANCZOS)
        # Resize ke ukuran baru
        resized_img = img.resize((new_size, new_size), Image.LANCZOS)
        resized_img.save(out_img_path)

    # Proses label: KOORDINAT TIDAK PERLU DIUBAH!
    # Karena format YOLO sudah normalized (0-1), resize gambar tidak mengubah koordinat normalized
    if os.path.exists(label_path):
        # Salin file label asli (tidak perlu modifikasi!)
        shutil.copy2(label_path, out_label_path)
    else:
        # Jika tidak ada label, buat file kosong
        with open(out_label_path, "w") as f:
            pass

    # Visualisasi bounding box pada gambar resized
    os.makedirs(os.path.dirname(out_bb_path), exist_ok=True)
    with Image.open(out_img_path) as img_bb:
        draw = ImageDraw.Draw(img_bb)
        if os.path.exists(out_label_path):
            with open(out_label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, bw, bh = parts
                    x, y, bw, bh = map(float, [x, y, bw, bh])
                    # Konversi ke pixel berdasarkan ukuran BARU
                    x_pixel = x * new_size
                    y_pixel = y * new_size
                    bw_pixel = bw * new_size
                    bh_pixel = bh * new_size
                    x_min = int(x_pixel - bw_pixel / 2)
                    y_min = int(y_pixel - bh_pixel / 2)
                    x_max = int(x_pixel + bw_pixel / 2)
                    y_max = int(y_pixel + bh_pixel / 2)
                    draw.rectangle([x_min, y_min, x_max, y_max],
                                   outline="red", width=2)
                    draw.text((x_min, y_min - 10), cls, fill="red")
        img_bb.save(out_bb_path)


# === Siapkan struktur folder baru ===
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(NEW_DATASET_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(NEW_DATASET_ROOT, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(NEW_DATASET_ROOT,
                "boundingbox", split), exist_ok=True)

# === Kumpulkan tugas ===
tasks = []
BB_ROOT = os.path.join(NEW_DATASET_ROOT, "boundingbox")

for split in ["train", "val", "test"]:
    orig_img_dir = os.path.join(ORIGINAL_DATASET_ROOT, "images", split)
    orig_label_dir = os.path.join(ORIGINAL_DATASET_ROOT, "labels", split)

    new_img_dir = os.path.join(NEW_DATASET_ROOT, "images", split)
    new_label_dir = os.path.join(NEW_DATASET_ROOT, "labels", split)
    bb_dir = os.path.join(BB_ROOT, split)

    for img_path in glob.glob(os.path.join(orig_img_dir, "*.*")):
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        label_path = os.path.join(orig_label_dir, name + ".txt")

        out_img_path = os.path.join(new_img_dir, filename)
        out_label_path = os.path.join(new_label_dir, name + ".txt")
        out_bb_path = os.path.join(bb_dir, filename)

        tasks.append((
            img_path, label_path, out_img_path, out_label_path, out_bb_path, NEW_SIZE, OLD_SIZE
        ))

# === Jalankan paralel ===
print(
    f"🔄 Memproses {len(tasks)} file, resize dari {OLD_SIZE}x{OLD_SIZE} ke {NEW_SIZE}x{NEW_SIZE}...")
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_file, *task) for task in tasks]
    for future in futures:
        future.result()

# Kumpulkan file bounding box untuk contoh
bb_files = []
for split in ["train", "val", "test"]:
    bb_files.extend(glob.glob(os.path.join(BB_ROOT, split, "*.*")))

print(f"✅ Selesai! Dataset resize tersimpan di: {NEW_DATASET_ROOT}")
print(f"📊 Total file: {len(bb_files)}")

# Tampilkan 20 contoh
samples = random.sample(bb_files, min(20, len(bb_files)))
print(f"\n📸 20 Contoh hasil bounding box (resize ke {NEW_SIZE}x{NEW_SIZE}):")
for s in samples:
    print(s)
