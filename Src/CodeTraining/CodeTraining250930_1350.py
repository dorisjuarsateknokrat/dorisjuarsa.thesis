import os
import glob
from datetime import datetime
from ultralytics import YOLO

# model_name = 'yolov8n.pt'
model_name = 'yolov8s.pt'
# model_name = 'yolov8m.pt'
# model_name = 'yolov8l.pt'
# model_name = 'yolov8x.pt'

model = YOLO(model_name)

# Path ke file data.yaml
dataset_path = '/home/dorisjuarsa/Public/DorisjuarsaProgramTrainingTesis/Datasets/DorisjuarsaDatasetBalance360x360RotateScale15_2509301304/data.yaml'

# Periksa apakah file dataset.yaml ada
if not os.path.exists(dataset_path):
    print(f"Error: File '{dataset_path}' tidak ditemukan.")
    print("Pastikan path ke file data.yaml sudah benar.")
    exit()

# === Ekstrak nama folder dataset untuk digunakan di 'name' ===
dataset_folder_name = os.path.basename(os.path.dirname(dataset_path))

# Tentukan singkatan model berdasarkan model yang dipilih (opsional tapi rapi)
if 'yolov8n.pt' in model_name:
    model_size = 'nano'
elif 'yolov8s.pt' in model_name:
    model_size = 'small'
elif 'yolov8m.pt' in model_name:
    model_size = 'medium'
elif 'yolov8l.pt' in model_name:
    model_size = 'large'
elif 'yolov8x.pt' in model_name:
    model_size = 'xl'
else:
    model_size = 'custom'  # fallback jika model tidak dikenali

# Buat nama run otomatis: <nama_folder_dataset>_<imgsz>_<model_size>_<timestamp>
imgsz = 416
timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
run_name = f"{dataset_folder_name}_{imgsz}_{model_size}_{timestamp}"

# === Bagian Tambahan untuk Pengecekan Dataset ===
print("\n--- Pengecekan Dataset ---")
base_dataset_dir = os.path.dirname(dataset_path)


def count_files(directory):
    if not os.path.exists(directory):
        return 0
    return len(glob.glob(os.path.join(directory, '*.*')))


print(f"Dataset root directory: {base_dataset_dir}")
print(
    f"Jumlah gambar training: {count_files(os.path.join(base_dataset_dir, 'images', 'train'))}")
print(
    f"Jumlah label training: {count_files(os.path.join(base_dataset_dir, 'labels', 'train'))}")
print(
    f"Jumlah gambar validation: {count_files(os.path.join(base_dataset_dir, 'images', 'val'))}")
print(
    f"Jumlah label validation: {count_files(os.path.join(base_dataset_dir, 'labels', 'val'))}")
print(
    f"Jumlah gambar test: {count_files(os.path.join(base_dataset_dir, 'images', 'test'))}")
print(
    f"Jumlah label test: {count_files(os.path.join(base_dataset_dir, 'labels', 'test'))}")
print("--- Pengecekan Dataset Selesai ---\n")

# Lanjutkan ke proses pelatihan
print("Memulai proses pelatihan YOLOv8...")

# Latih model
results = model.train(
    data=dataset_path,
    epochs=100,
    imgsz=imgsz,
    device=0,
    batch=128,
    name=run_name,
    patience=20,
    cos_lr=True,
    cache=False,
    workers=4,

    # === MATIKAN SEMUA AUGMENTASI ===
    hsv_h=0.0,      # hue
    hsv_s=0.0,      # saturation
    hsv_v=0.0,      # value/brightness
    degrees=0.0,    # rotation
    translate=0.0,  # translation
    scale=0.0,      # scaling (zoom)
    shear=0.0,      # shear
    perspective=0.0,  # perspective warp
    flipud=0.0,     # vertical flip
    fliplr=0.0,     # horizontal flip
    mosaic=0.0,     # mosaic augmentation
    mixup=0.0,      # mixup
    copy_paste=0.0  # copy-paste augmentation
)

print(f"Pelatihan selesai. Hasil disimpan di folder 'runs/detect/{run_name}'.")
