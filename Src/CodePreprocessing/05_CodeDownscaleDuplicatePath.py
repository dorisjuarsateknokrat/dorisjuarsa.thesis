import os
import glob
import random
import shutil
from datetime import datetime
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor

# === KONFIGURASI ===
ORIGINAL_DATASET_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/DorisjuarsaDatasetBalance360x360Rotate"

# Parameter downscaling
SCALE_FACTOR = 0.15  # 50%
SCALE_PERCENT = int(SCALE_FACTOR * 100)

# Buat folder output baru (tanpa menyalin file asli)
timestamp = datetime.now().strftime("%y%m%d%H%M")
NEW_DATASET_ROOT = f"{ORIGINAL_DATASET_ROOT}Scale{SCALE_PERCENT}_{timestamp}"

print(f"📁 Folder dataset hasil downscaling: {NEW_DATASET_ROOT}")

# Fungsi transformasi


def process_file(img_path, label_path, out_img_path, out_label_path, out_bb_path, scale_factor, img_size=(360, 360)):
    w, h = img_size

    # Proses gambar
    with Image.open(img_path) as img:
        if img.size != img_size:
            img = img.resize(img_size, Image.LANCZOS)
        # Resize konten
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img_scaled = img.resize((new_w, new_h), Image.LANCZOS)
        # Canvas putih
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        canvas.paste(img_scaled, (paste_x, paste_y))
        canvas.save(out_img_path)

    # Proses label
    new_lines = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_norm, y_norm, bw_norm, bh_norm = parts
                x_norm, y_norm, bw_norm, bh_norm = map(
                    float, [x_norm, y_norm, bw_norm, bh_norm])

                # Ke pixel
                x_px = x_norm * w
                y_px = y_norm * h
                bw_px = bw_norm * w
                bh_px = bh_norm * h

                # Transformasi terhadap pusat
                cx_img, cy_img = w / 2, h / 2
                x_new_px = cx_img + (x_px - cx_img) * scale_factor
                y_new_px = cy_img + (y_px - cy_img) * scale_factor
                bw_new_px = bw_px * scale_factor
                bh_new_px = bh_px * scale_factor

                # Normalisasi ulang
                x_new_norm = x_new_px / w
                y_new_norm = y_new_px / h
                bw_new_norm = bw_new_px / w
                bh_new_norm = bh_new_px / h

                # Clamp ke [0,1] (opsional tapi aman)
                x_new_norm = max(0.0, min(1.0, x_new_norm))
                y_new_norm = max(0.0, min(1.0, y_new_norm))
                bw_new_norm = max(0.0, min(1.0, bw_new_norm))
                bh_new_norm = max(0.0, min(1.0, bh_new_norm))

                new_lines.append(
                    f"{cls} {x_new_norm:.6f} {y_new_norm:.6f} {bw_new_norm:.6f} {bh_new_norm:.6f}\n")

        with open(out_label_path, "w") as f:
            f.writelines(new_lines)

    # Visualisasi bounding box
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
                    x_pixel = x * w
                    y_pixel = y * h
                    bw_pixel = bw * w
                    bh_pixel = bh * h
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

        # Simpan dengan NAMA SAMA (tidak ada suffix)
        out_img_path = os.path.join(new_img_dir, filename)
        out_label_path = os.path.join(new_label_dir, name + ".txt")
        out_bb_path = os.path.join(bb_dir, filename)

        tasks.append((
            img_path, label_path, out_img_path, out_label_path, out_bb_path, SCALE_FACTOR
        ))

# === Jalankan paralel ===
print(
    f"🔄 Memproses {len(tasks)} file dengan downscaling {SCALE_FACTOR*100:.0f}%...")
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_file, *task) for task in tasks]
    for future in futures:
        future.result()  # Tunggu semua selesai

# Kumpulkan file bounding box untuk contoh
bb_files = []
for split in ["train", "val", "test"]:
    bb_files.extend(glob.glob(os.path.join(BB_ROOT, split, "*.*")))

print(f"✅ Selesai! Dataset downscaling tersimpan di: {NEW_DATASET_ROOT}")
print(f"📊 Total file: {len(bb_files)}")

# Tampilkan 20 contoh
samples = random.sample(bb_files, min(20, len(bb_files)))
print("\n📸 20 Contoh hasil bounding box (downscaling penuh):")
for s in samples:
    print(s)
