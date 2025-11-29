# augment_rotate_copy.py
import cv2
import numpy as np
import shutil
import sys
from pathlib import Path

SRC = Path(
    "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/DorisjuarsaDatasetBalance360x360")
DST = SRC.parent / "DorisjuarsaDatasetBalance360x360Rotate"


def rotate_yolo_label(txt_path, deg, img_wh):
    w, h = img_wh
    boxes = []
    with open(txt_path) as f:
        for line in f:
            cls, x_c, y_c, bw, bh = map(float, line.split())
            boxes.append([cls, x_c, y_c, bw, bh])

    new_boxes = []
    for cls, x_c, y_c, bw, bh in boxes:
        x1 = (x_c - bw/2) * w
        y1 = (y_c - bh/2) * h
        x2 = (x_c + bw/2) * w
        y2 = (y_c + bh/2) * h

        if deg == 90:
            x1n, y1n, x2n, y2n = y1, w - x2, y2, w - x1
            wn, hn = h, w
        elif deg == 180:
            x1n, y1n, x2n, y2n = w - x2, h - y2, w - x1, h - y1
            wn, hn = w, h
        else:  # 270
            x1n, y1n, x2n, y2n = h - y2, x1, h - y1, x2
            wn, hn = h, w

        bw_n = (x2n - x1n) / wn
        bh_n = (y2n - y1n) / hn
        xc_n = ((x1n + x2n)/2) / wn
        yc_n = ((y1n + y2n)/2) / hn
        new_boxes.append([cls, xc_n, yc_n, bw_n, bh_n])
    return new_boxes


def copy_then_augment():
    # 1. salin seluruh struktur (images + labels + data.yaml)
    if DST.exists():
        print("Folder tujuan sudah ada – batalkan atau hapus manual.")
        sys.exit(1)
    shutil.copytree(SRC, DST, dirs_exist_ok=False)

    # 2. augmentasi di folder salinan
    for split in ("train", "val", "test"):
        img_dir = DST / "images" / split
        lbl_dir = DST / "labels" / split
        for img_path in list(img_dir.glob("*.jpg")):   # list → agar baru bisa ditulis
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            for deg in (90, 180, 270):
                # gambar
                if deg == 90:
                    rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif deg == 180:
                    rot = cv2.rotate(img, cv2.ROTATE_180)
                else:
                    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(str(img_dir / f"{img_path.stem}_{deg}.jpg"), rot)

                # label
                txt = lbl_dir / f"{img_path.stem}.txt"
                if txt.exists():
                    boxes = rotate_yolo_label(txt, deg, (w, h))
                    with open(lbl_dir / f"{img_path.stem}_{deg}.txt", "w") as f:
                        for b in boxes:
                            f.write(" ".join(map(str, b)) + "\n")
                else:
                    (lbl_dir / f"{img_path.stem}_{deg}.txt").touch()

    print("Selesai. Dataset rotasi ada di:", DST)


if __name__ == "__main__":
    copy_then_augment()
