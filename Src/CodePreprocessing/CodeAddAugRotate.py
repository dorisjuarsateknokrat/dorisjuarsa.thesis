import os
import glob
import random
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor

# Root dataset hasil resize 360x360
DATASET_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/datasetDorisjuarsaBalance_Undersampling_50pct_20250927_1426_360x360_20250927_1434"

# Folder boundingbox
BB_ROOT = os.path.join(DATASET_ROOT, "boundingbox")

# Fungsi rotasi + update label


def rotate_180(img_path, label_path, out_img_path, out_label_path, out_bb_path):
    with Image.open(img_path) as img:
        w, h = img.size
        # rotate 180 derajat
        img_rot = img.rotate(180)
        img_rot.save(out_img_path)

    # update label
    new_lines = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, bw, bh = parts
                x, y, bw, bh = map(float, [x, y, bw, bh])
                # update koordinat (YOLO normalisasi)
                x_new = 1 - x
                y_new = 1 - y
                new_lines.append(
                    f"{cls} {x_new:.6f} {y_new:.6f} {bw:.6f} {bh:.6f}\n")

        with open(out_label_path, "w") as f:
            f.writelines(new_lines)

    # buat versi bounding box
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
                                   outline="blue", width=2)
                    draw.text((x_min, y_min - 10), cls, fill="blue")
        img_bb.save(out_bb_path)


# Kumpulkan semua tugas terlebih dahulu
tasks = []
for split in ["train", "val", "test"]:
    img_dir = os.path.join(DATASET_ROOT, "images", split)
    label_dir = os.path.join(DATASET_ROOT, "labels", split)
    bb_dir = os.path.join(BB_ROOT, split)
    os.makedirs(bb_dir, exist_ok=True)

    for img_path in glob.glob(os.path.join(img_dir, "*.*")):
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        label_path = os.path.join(label_dir, name + ".txt")

        out_img_path = os.path.join(img_dir, f"{name}_180{ext}")
        out_label_path = os.path.join(label_dir, f"{name}_180.txt")
        out_bb_path = os.path.join(bb_dir, f"{name}_180{ext}")

        # supaya tidak duplikat kalau dijalankan ulang
        if not os.path.exists(out_img_path):
            tasks.append((img_path, label_path, out_img_path,
                         out_label_path, out_bb_path))

# Jalankan secara paralel
augmented_files = []
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(rotate_180, *task) for task in tasks]
    # Tunggu semua selesai dan kumpulkan hasil (nama file bounding box)
    for task, future in zip(tasks, futures):
        future.result()  # opsional: untuk menangkap exception jika ada
        # out_bb_path adalah elemen ke-5 (index 4)
        augmented_files.append(task[4])

print(f"✅ Augmentasi 180° selesai. Total file baru: {len(augmented_files)}")

# Ambil 20 sample random bounding box hasil augmentasi
if len(augmented_files) > 20:
    samples = random.sample(augmented_files, 20)
else:
    samples = augmented_files

print("\n📸 20 Contoh hasil bounding box (augmentasi 180°):")
for s in samples:
    print(s)
