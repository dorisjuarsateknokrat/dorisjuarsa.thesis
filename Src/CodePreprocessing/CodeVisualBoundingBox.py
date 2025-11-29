# visualize_yolo.py
import cv2
import yaml
from pathlib import Path
import random

DATASET = Path(
    "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/DorisjuarsaDatasetBalance360x360_rotate")
VIS_ROOT = DATASET.parent / "DorisjuarsaDatasetBalance360x360_rotate_VIS"

# ---------- warna per kelas ----------
with open(DATASET / "data.yaml") as f:
    cfg = yaml.safe_load(f)
id2name = cfg["names"]          # {0:'BAS', 1:'EOS', ...}
random.seed(42)
colors = {i: tuple(random.randint(0, 255) for _ in range(3))
          for i in id2name.keys()}


def draw_yolo(img_path, txt_path, dst_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    if txt_path.exists():
        with open(txt_path) as f:
            for line in f:
                cls, x_c, y_c, bw, bh = map(float, line.split())
                # YOLO → pixel
                x1 = int((x_c - bw/2) * w)
                y1 = int((y_c - bh/2) * h)
                x2 = int((x_c + bw/2) * w)
                y2 = int((y_c + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), colors[int(cls)], 2)
                cv2.putText(img, id2name[int(cls)], (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[int(cls)], 2)
    cv2.imwrite(str(dst_path), img)


def vis_all():
    for split in ("train", "val", "test"):
        src_img = DATASET / "images" / split
        src_lbl = DATASET / "labels" / split
        dst_img = VIS_ROOT / split
        dst_img.mkdir(parents=True, exist_ok=True)

        for img_p in src_img.glob("*.jpg"):
            txt_p = src_lbl / f"{img_p.stem}.txt"
            draw_yolo(img_p, txt_p, dst_img / img_p.name)

    print("Selesai. Cek hasil di:", VIS_ROOT)


if __name__ == "__main__":
    vis_all()
