from pathlib import Path
from ultralytics import YOLO

# ==================================================
# BASE PATH
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# DATASET CONFIG
# ==================================================
DATASETS_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "Datasets"
).resolve()

MODEL_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "DataModels"
).resolve()

YOLO_DATASET_DIR = DATASETS_DIR / "DorisjuarsaDatasetYoloBaseSizeToScale0_25Clahe"
DATASET_YAML = YOLO_DATASET_DIR / "data.yaml"

if not DATASET_YAML.exists():
    raise FileNotFoundError(f"❌ data.yaml tidak ditemukan:\n{DATASET_YAML}")

# ==================================================
# RUNS PATH (BISA KAMU ATUR)
# ==================================================
RUNS_ROOT = MODEL_DIR / "runs"      # <<==== GANTI DI SINI JIKA PERLU
RUNS_TASK = "detect"                   # detect / segment / classify

# ==================================================
# TRAINING CONFIG
# ==================================================
imgsz = 640
epochs = 60
batch = 32
device = 0

model_name = "yolov8s.pt"

# ==================================================
# RUN NAME (KONSISTEN & RAPI)
# ==================================================
dataset_name = YOLO_DATASET_DIR.name

if "yolov8n" in model_name:
    model_size = "nano"
elif "yolov8s" in model_name:
    model_size = "small"
elif "yolov8m" in model_name:
    model_size = "medium"
elif "yolov8l" in model_name:
    model_size = "large"
elif "yolov8x" in model_name:
    model_size = "xl"
else:
    model_size = "custom"

run_name = f"{dataset_name}_{imgsz}_{model_size}"

# ==================================================
# AUTO RESUME LOGIC (PALING AMAN)
# ==================================================
RUN_DIR = RUNS_ROOT / RUNS_TASK / run_name
WEIGHTS_DIR = RUN_DIR / "weights"
LAST_CKPT = WEIGHTS_DIR / "last.pt"

if LAST_CKPT.exists():
    print("♻️ MODE: RESUME TRAINING")
    print(f"   ↳ checkpoint: {LAST_CKPT}")
    model = YOLO(LAST_CKPT)
else:
    print("🆕 MODE: TRAINING BARU")
    print(f"   ↳ pretrained: {model_name}")
    model = YOLO(model_name)

# ==================================================
# SANITY CHECK DATASET
# ==================================================


def count_files(p):
    return len(list(p.glob("*.*"))) if p.exists() else 0


print("\n--- DATASET CHECK ---")
print(f"Dataset root : {YOLO_DATASET_DIR}")
print(f"Train images : {count_files(YOLO_DATASET_DIR / 'images' / 'train')}")
print(f"Train labels : {count_files(YOLO_DATASET_DIR / 'labels' / 'train')}")
print(f"Val images   : {count_files(YOLO_DATASET_DIR / 'images' / 'val')}")
print(f"Val labels   : {count_files(YOLO_DATASET_DIR / 'labels' / 'val')}")
print(f"Test images  : {count_files(YOLO_DATASET_DIR / 'images' / 'test')}")
print(f"Test labels  : {count_files(YOLO_DATASET_DIR / 'labels' / 'test')}")
print("---------------------\n")

# ==================================================
# TRAINING
# ==================================================
print("🚀 Training YOLOv8 dimulai...\n")

results = model.train(
    data=str(DATASET_YAML),
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    device=device,

    project=str(RUNS_ROOT),
    name=run_name,

    patience=20,
    cos_lr=True,
    cache=False,
    workers=4,

    # ===============================
    # AUGMENTATION (AMANKAN)
    # ===============================
    fliplr=0.5,
    degrees=10.0,
    translate=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    mixup=0.0,
    copy_paste=0.0,
)

print("\n✅ Training selesai / dilanjutkan.")
print(f"📁 Output tersimpan di:\n{RUN_DIR}")
