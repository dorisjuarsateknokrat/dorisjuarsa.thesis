from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# ==================================================
# BASE PATH
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# DATASETS ROOT
# ==================================================
DATASETS_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "Datasets"
).resolve()

# ==================================================
# DATA MODELS ROOT
# ==================================================
MODELS_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "DataModels"
).resolve()

# ==================================================
# OUTPUT DIR
# ==================================================
OUTPUT_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "DataTesting"
    / "Output"
    / "YOLOv8_Comparison"
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# MODEL PATHS
# ==================================================
MODEL_STANDARD = (
    MODELS_DIR
    / "runs"
    / "DorisjuarsaDatasetYoloBaseSizeToScale0_25_640_small"
    / "weights"
    / "best.pt"
)

MODEL_CLAHE = (
    MODELS_DIR
    / "runs"
    / "DorisjuarsaDatasetYoloBaseSizeToScale0_25Clahe_640_small"
    / "weights"
    / "best.pt"
)

# ==================================================
# DATASET PATHS
# ==================================================
DATA_STD = DATASETS_DIR / "DorisjuarsaDatasetYoloBaseSizeToScale0_25"
DATA_CLAHE = DATASETS_DIR / "DorisjuarsaDatasetYoloBaseSizeToScale0_25Clahe"

# ==================================================
# SCENARIO CONFIG
# ==================================================
SCENARIOS = [
    ("ModelStd_DataStd", MODEL_STANDARD, DATA_STD),
    ("ModelStd_DataClahe", MODEL_STANDARD, DATA_CLAHE),
    ("ModelClahe_DataStd", MODEL_CLAHE, DATA_STD),
    ("ModelClahe_DataClahe", MODEL_CLAHE, DATA_CLAHE),
]

# ==================================================
# EVALUATION
# ==================================================
results_summary = []

for name, model_path, data_root in SCENARIOS:
    print(f"\n🚀 Evaluating: {name}")

    model = YOLO(model_path)

    metrics = model.val(
        data=data_root / "data.yaml",
        split="test",
        imgsz=640,
        conf=0.25,
        iou=0.5,
        plots=True,
        project=OUTPUT_DIR,
        name=name
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    results_summary.append({
        "Scenario": name,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "mAP@0.5": round(map50, 4),
        "mAP@0.5:0.95": round(map5095, 4)
    })

# ==================================================
# SAVE SUMMARY CSV
# ==================================================
df = pd.DataFrame(results_summary)
csv_path = OUTPUT_DIR / "YOLOv8_4Scenario_Comparison.csv"
df.to_csv(csv_path, index=False)

print("\n✅ Evaluation completed!")
print(f"📄 Summary saved to: {csv_path}")
