#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microscope GUI - Full Feature v2
- Two 720x720 canvases (left original w/ pan+zoom, right annotated)
- Load model, open media (image/video), open camera, detect folder, reset, stop
- Folder outputs to ../OutputTesting/<timestamp>_<foldername>/ and CSV in same folder
Author: adapted for Doris Juarsa (final thesis)
"""

import sys
import os
import time
import csv
from datetime import datetime
import threading
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QMessageBox, QHBoxLayout, QVBoxLayout, QProgressBar, QStatusBar,
    QInputDialog, QSizePolicy, QFrame, QCheckBox
)

# ---------------- CONFIG ----------------
SQUARE = 720
GAP = 5
WINDOW_WIDTH = SQUARE * 2 + GAP + 40
WINDOW_HEIGHT = SQUARE + 160
DEFAULT_CAMERA_W = 1280
DEFAULT_CAMERA_H = 720
DEFAULT_CONF = 0.25  # confidence threshold for inference
BUTTON_WIDTH = 190
BUTTON_HEIGHT = 40


# ---------------- Helpers ----------------
def qpixmap_from_ndarray_rgb(img_rgb: np.ndarray) -> QPixmap:
    if img_rgb is None:
        return QPixmap()
    h, w, ch = img_rgb.shape
    return QPixmap.fromImage(QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy())


def center_crop_square(img_bgr: np.ndarray, size=SQUARE) -> np.ndarray:
    if img_bgr is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    if w == h == size:
        return img_bgr.copy()
    cx, cy = w // 2, h // 2
    half = size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = x1 + size, y1 + size
    if x2 > w:
        x2, x1 = w, max(0, w - size)
    if y2 > h:
        y2, y1 = h, max(0, h - size)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def label_color(name: str):
    h = abs(hash(name))
    return (int(h % 200) + 30, int((h // 200) % 200) + 30, int((h // 40000) % 200) + 30)


# ---------------- ImageLabel (Left & Right canvas) ----------------
class ImageLabel(QLabel):
    view_changed = Signal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.setFixedSize(SQUARE, SQUARE)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #000; border: 1px solid #444;")
        self._pixmap = None
        self._scale = 1.0
        self._offset = [0.0, 0.0]
        self._dragging = False
        self._last_pos = None
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_image_rgb(self, img_rgb: np.ndarray, reset=False):
        if img_rgb is None:
            self._pixmap = None
            self.setPixmap(QPixmap())
            return
        pm = qpixmap_from_ndarray_rgb(img_rgb)
        if reset:
            self._scale = 1.0
            self._offset = [0.0, 0.0]
        self._pixmap = pm
        self._update_display()
        self.emit_view_changed()

    def _update_display(self):
        if self._pixmap is None:
            self.setPixmap(QPixmap())
            return
        orig_w, orig_h = self._pixmap.width(), self._pixmap.height()
        scaled = self._pixmap.scaled(int(orig_w * self._scale), int(orig_h * self._scale),
                                     Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        canvas = QPixmap(self.width(), self.height())
        canvas.fill(Qt.GlobalColor.black)
        painter = QPainter(canvas)
        painter.drawPixmap(int(self._offset[0]), int(self._offset[1]), scaled)
        painter.end()
        self.setPixmap(canvas)

    def wheelEvent(self, event):
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 0.85
        self._scale = max(0.2, min(self._scale * factor, 6.0))
        # update offset so zoom centers on cursor roughly (simple behavior)
        pos = event.position()
        mx, my = pos.x(), pos.y()
        old_img_x = (mx - self._offset[0]) / (self._scale / factor)
        old_img_y = (my - self._offset[1]) / (self._scale / factor)
        self._offset[0] = mx - old_img_x * self._scale
        self._offset[1] = my - old_img_y * self._scale
        self._update_display()
        self.emit_view_changed()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._dragging = True
            self._last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self._dragging and self._pixmap is not None:
            pos = event.position()
            dx = pos.x() - self._last_pos.x()
            dy = pos.y() - self._last_pos.y()
            self._offset[0] += dx
            self._offset[1] += dy
            self._last_pos = pos
            self._update_display()
            self.emit_view_changed()

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def emit_view_changed(self):
        if self._pixmap is None:
            return
        img_w = self._pixmap.width()
        img_h = self._pixmap.height()
        x1 = int(round((0 - self._offset[0]) / self._scale))
        y1 = int(round((0 - self._offset[1]) / self._scale))
        x2 = int(round((self.width() - self._offset[0]) / self._scale))
        y2 = int(round((self.height() - self._offset[1]) / self._scale))
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, img_w, img_h
        self.view_changed.emit(x1, y1, x2, y2)

    def set_view(self, scale, offset_x, offset_y):
        if self._pixmap is None:
            return
        self._scale = scale
        self._offset = [offset_x, offset_y]
        self._update_display()


# ---------------- Worker Thread (Inference) ----------------
class InferenceWorker(QThread):
    # orig_rgb, annotated_rgb, counts, time
    frame_processed = Signal(np.ndarray, np.ndarray, dict, float)
    folder_progress = Signal(int, int)
    folder_finished = Signal(str, float, float)
    error = Signal(str)

    def __init__(self, model: YOLO, device: str = 'cpu', use_clahe: bool = False, clahe_clip: float = 2.0, clahe_tile: int = 8):
        super().__init__()
        self.model = model
        self.device = device
        self._mode = None
        self._running = False
        self._input_path = None
        self._cam_index = None
        self._folder = None
        self._out_folder = None
        self._crop_rect = None
        self._trigger = threading.Event()

        # CLAHE config
        self.use_clahe = bool(use_clahe)
        self.clahe_clip = float(clahe_clip)
        self.clahe_tile = int(clahe_tile)

    def configure_image(self, path):
        self._mode = 'image'
        self._input_path = path

    def configure_video(self, path):
        self._mode = 'video'
        self._input_path = path

    def configure_camera(self, index):
        self._mode = 'camera'
        self._cam_index = index

    def configure_folder(self, folder, out_folder):
        self._mode = 'folder'
        self._folder = folder
        self._out_folder = out_folder

    def set_crop_rect(self, rect):
        # rect in image coordinate of the loaded pixmap (x1,y1,x2,y2)
        self._crop_rect = rect

    def trigger_inference(self):
        self._trigger.set()

    def stop(self):
        self._running = False
        self._trigger.set()

    def _infer(self, bgr_img):
        t0 = time.time()
        results = self.model(bgr_img, device=self.device,
                             verbose=False, conf=DEFAULT_CONF)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return results, time.time() - t0

    def _apply_clahe(self, bgr_img):
        """
        Apply CLAHE on L channel of LAB, return BGR image.
        """
        try:
            lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip, tileGridSize=(
                    self.clahe_tile, self.clahe_tile)
            )
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            # fallback: return original if something goes wrong
            return bgr_img

    def _draw_boxes(self, base_bgr, results):
        ann = base_bgr.copy()
        res = results[0].to('cpu')
        boxes = getattr(res, 'boxes', None)
        dets = []
        if boxes is None or len(boxes) == 0:
            return ann, dets
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = res.names
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            label = names.get(int(cls), str(cls))
            col = label_color(label)
            # rectangle
            cv2.rectangle(ann, (x1i, y1i), (x2i, y2i), col, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # label background
            by = max(0, y1i - th - 6)
            cv2.rectangle(ann, (x1i, by), (x1i + tw + 6, by + th + 4), col, -1)
            cv2.putText(ann, text, (x1i + 3, by + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            dets.append((label, float(conf), x1i, y1i, x2i, y2i))
        return ann, dets

    def _counts_from_results(self, results):
        res = results[0].to('cpu')
        boxes = getattr(res, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return {}
        labels = boxes.cls.cpu().numpy().astype(int)
        names = res.names
        counts = {}
        for l in labels:
            n = names.get(int(l), str(l))
            counts[n] = counts.get(n, 0) + 1
        return counts

    def run(self):
        if self.model is None:
            self.error.emit("Model belum dimuat.")
            return
        self._running = True
        try:
            if self._mode == 'camera':
                cap = cv2.VideoCapture(
                    self._cam_index if self._cam_index is not None else 0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_H)
                if not cap.isOpened():
                    self.error.emit("Gagal membuka kamera.")
                    return
                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_sq = center_crop_square(frame, SQUARE)
                    # keep original RGB for left display
                    orig_rgb = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
                    # apply crop_rect if any (left canvas may set)
                    rect = self._crop_rect
                    if rect:
                        x1, y1, x2, y2 = rect
                        crop = frame_sq[y1:y2, x1:x2].copy()
                        img_for_infer = crop if crop.size else frame_sq
                        offset_x, offset_y = x1, y1
                    else:
                        img_for_infer = frame_sq
                        offset_x, offset_y = 0, 0

                    # apply CLAHE to the image passed to the model (if enabled)
                    if self.use_clahe:
                        img_for_infer_proc = self._apply_clahe(img_for_infer)
                    else:
                        img_for_infer_proc = img_for_infer

                    results, dt = self._infer(img_for_infer_proc)
                    ann_bgr, dets = self._draw_boxes(
                        img_for_infer_proc, results)
                    # compose full annotated if offset
                    if offset_x != 0 or offset_y != 0:
                        full_annot = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
                        ah, aw = ann_bgr.shape[:2]
                        full_annot[offset_y:offset_y+ah, offset_x:offset_x +
                                   aw] = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        full_annot = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                    counts = self._counts_from_results(results)
                    self.frame_processed.emit(orig_rgb, full_annot, counts, dt)
                    QThread.msleep(30)
                cap.release()

            elif self._mode == 'video':
                cap = cv2.VideoCapture(self._input_path)
                if not cap.isOpened():
                    self.error.emit("Gagal membuka video.")
                    return
                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_sq = center_crop_square(frame, SQUARE)
                    orig_rgb = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
                    rect = self._crop_rect
                    if rect:
                        x1, y1, x2, y2 = rect
                        crop = frame_sq[y1:y2, x1:x2].copy()
                        img_for_infer = crop if crop.size else frame_sq
                        offset_x, offset_y = x1, y1
                    else:
                        img_for_infer = frame_sq
                        offset_x, offset_y = 0, 0

                    if self.use_clahe:
                        img_for_infer_proc = self._apply_clahe(img_for_infer)
                    else:
                        img_for_infer_proc = img_for_infer

                    results, dt = self._infer(img_for_infer_proc)
                    ann_bgr, dets = self._draw_boxes(
                        img_for_infer_proc, results)
                    if offset_x != 0 or offset_y != 0:
                        full_annot = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
                        ah, aw = ann_bgr.shape[:2]
                        full_annot[offset_y:offset_y+ah, offset_x:offset_x +
                                   aw] = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        full_annot = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                    counts = self._counts_from_results(results)
                    self.frame_processed.emit(orig_rgb, full_annot, counts, dt)
                    QThread.msleep(30)
                cap.release()

            elif self._mode == 'image':
                img = cv2.imread(self._input_path)
                if img is None:
                    self.error.emit("Gagal membaca gambar.")
                    return
                frame_sq = center_crop_square(img, SQUARE)
                orig_rgb = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
                rect = self._crop_rect
                if rect:
                    x1, y1, x2, y2 = rect
                    crop = frame_sq[y1:y2, x1:x2].copy()
                    img_for_infer = crop if crop.size else frame_sq
                    offset_x, offset_y = x1, y1
                else:
                    img_for_infer = frame_sq
                    offset_x, offset_y = 0, 0

                if self.use_clahe:
                    img_for_infer_proc = self._apply_clahe(img_for_infer)
                else:
                    img_for_infer_proc = img_for_infer

                results, dt = self._infer(img_for_infer_proc)
                ann_bgr, dets = self._draw_boxes(img_for_infer_proc, results)
                if offset_x != 0 or offset_y != 0:
                    full_annot = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
                    ah, aw = ann_bgr.shape[:2]
                    full_annot[offset_y:offset_y+ah, offset_x:offset_x +
                               aw] = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                else:
                    full_annot = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                counts = self._counts_from_results(results)
                self.frame_processed.emit(orig_rgb, full_annot, counts, dt)

                # wait for manual triggers (zoom/pan) while in image mode
                while self._running:
                    self._trigger.wait(timeout=0.5)
                    if not self._running:
                        break
                    if self._trigger.is_set():
                        self._trigger.clear()
                        rect = self._crop_rect
                        if rect:
                            x1, y1, x2, y2 = rect
                            crop = frame_sq[y1:y2, x1:x2].copy()
                            img_for_infer = crop if crop.size else frame_sq
                            offset_x, offset_y = x1, y1
                        else:
                            img_for_infer = frame_sq
                            offset_x, offset_y = 0, 0

                        if self.use_clahe:
                            img_for_infer_proc = self._apply_clahe(
                                img_for_infer)
                        else:
                            img_for_infer_proc = img_for_infer

                        results, dt = self._infer(img_for_infer_proc)
                        ann_bgr, dets = self._draw_boxes(
                            img_for_infer_proc, results)
                        if offset_x != 0 or offset_y != 0:
                            full_annot = cv2.cvtColor(
                                frame_sq, cv2.COLOR_BGR2RGB)
                            ah, aw = ann_bgr.shape[:2]
                            full_annot[offset_y:offset_y+ah, offset_x:offset_x +
                                       aw] = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                        else:
                            full_annot = cv2.cvtColor(
                                ann_bgr, cv2.COLOR_BGR2RGB)
                        counts = self._counts_from_results(results)
                        self.frame_processed.emit(
                            orig_rgb, full_annot, counts, dt)

            elif self._mode == 'folder':
                files = [f for f in os.listdir(self._folder) if f.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.bmp'))]
                total = len(files)
                if total == 0:
                    self.error.emit("Tidak ada file gambar di folder.")
                    return
                os.makedirs(self._out_folder, exist_ok=True)
                all_dets = []
                start_total = time.time()
                for i, fname in enumerate(files, start=1):
                    if not self._running:
                        break
                    fpath = os.path.join(self._folder, fname)
                    img = cv2.imread(fpath)
                    if img is None:
                        self.folder_progress.emit(i, total)
                        continue
                    frame_sq = center_crop_square(img, SQUARE)

                    # For folder processing: apply CLAHE to the image that will be passed to model
                    if self.use_clahe:
                        frame_proc = self._apply_clahe(frame_sq)
                    else:
                        frame_proc = frame_sq

                    results, dt = self._infer(frame_proc)
                    ann_bgr, dets = self._draw_boxes(frame_proc, results)

                    # save annotated image (BGR expected by imwrite)
                    cv2.imwrite(os.path.join(self._out_folder, fname), ann_bgr)

                    # collect dets for CSV (use dets list)
                    if dets:
                        base_name = os.path.splitext(fname)[0]
                        file_prefix = base_name[:3].lower()
                        for label, conf, x1, y1, x2, y2 in dets:
                            cls_pref = label[:3].lower()
                            match = (file_prefix == cls_pref)
                            all_dets.append([fname, label, round(
                                conf, 4), x1, y1, x2, y2, match, round(dt, 6)])

                    self.folder_progress.emit(i, total)

                total_time = time.time() - start_total
                avg = total_time / total if total else 0.0
                csv_path = os.path.join(
                    self._out_folder, "detection_results.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['filename', 'class_name', 'confidence', 'x_min',
                                    'y_min', 'x_max', 'y_max', 'match_label', 'time_sec'])
                    writer.writerows(all_dets)
                self.folder_finished.emit(csv_path, total_time, avg)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._running = False


# ---------------- Main Window ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DORIS JUARSA - 24321025")
        # self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # Tetapkan lebar tetap, tinggi otomatis menyesuaikan isi layout
        self.setFixedWidth(WINDOW_WIDTH)

        # ---------- STATE ----------
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.worker = None

        # CLAHE defaults (sesuai preview app Anda)
        self.clahe_clip = 2.0
        self.clahe_tile = 8

        # ---------- UI: CANVASES ----------
        self.left = ImageLabel()
        self.right = ImageLabel()
        self.left.view_changed.connect(self.on_view_changed)

        # ---------- BANNER ----------
        banner_text = """
        <div style="text-align:center;">
            <div style="font-size:16pt;">Model Deteksi Objek Real-Time Untuk Klasifikasi Otomatis Subtipe Leukosit Pada Citra Mikroskopis Darah Dengan YOLOv8</div>
            <div style="font-size:11pt; margin-top:8px;">Dibuat oleh: Doris Juarsa</div>
            <div style="font-size:11pt; margin-top:8px;">NPM: 24321025</div>
            <div style="font-size:11pt; margin-top:4px;">Pembimbing: Dr. Erliyan Redy Susanto, S.Kom., M.Kom.</div>
            <div style="font-size:11pt; margin-top:4px;">Penguji 1: Dr. Rohmat Indra Borman, M.Kom.</div>
        </div>
        """

        # banner_text = "DUMMY"
        self.banner = QLabel()
        self.banner.setText(banner_text)
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 12px;
        """)

        # ---------- CONTROL BUTTONS ----------
        self.btn_load = QPushButton("Load Model")
        self.btn_open_media = QPushButton("Open Media")
        self.btn_open_cam = QPushButton("Open Camera")
        self.btn_detect_folder = QPushButton("Detect Folder")
        self.btn_reset = QPushButton("Reset Position")
        self.btn_stop = QPushButton("Stop")

        # Button CLAHE (added minimally)
        self.btn_clahe = QPushButton("CLAHE: OFF")
        self.btn_clahe.setCheckable(True)
        # self.btn_clahe.setStyleSheet(
        #     "background-color: #ccc; font-weight: bold; padding: 6px;")
        self.btn_clahe.setChecked(False)
        self.btn_clahe.clicked.connect(self.on_clahe_button_toggled)

        # disable buttons sebelum model dimuat
        for b in [self.btn_open_media, self.btn_open_cam, self.btn_detect_folder, self.btn_reset, self.btn_stop]:
            b.setEnabled(False)

        # Tombol layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        for b in [self.btn_load, self.btn_open_media, self.btn_open_cam, self.btn_detect_folder, self.btn_reset, self.btn_stop]:
            buttons_layout.addWidget(b)

        # ---------- STYLE & SIZE BUTTONS ----------
        for b in [self.btn_load, self.btn_open_media, self.btn_open_cam,
                  self.btn_detect_folder, self.btn_reset, self.btn_stop, self.btn_clahe]:
            b.setFixedWidth(BUTTON_WIDTH)
            b.setFixedHeight(BUTTON_HEIGHT)
            # b.setStyleSheet("""
            #     QPushButton {
            #         background-color: #e0e0e0;
            #         border: 1px solid #aaa;
            #         border-radius: 6px;
            #         font-weight: bold;
            #     }
            #     QPushButton:hover {
            #         background-color: #d6d6d6;
            #     }
            #     QPushButton:checked {
            #         background-color: #a8e6a2;  /* efek hijau lembut untuk tombol CLAHE ON */
            #     }
            # """)

        # tambahkan checkbox ke layout tombol (tidak mengubah tombol existing)
        buttons_layout.addWidget(self.btn_clahe)

        # Bungkus tombol agar bisa diberi background seperti “div”
        buttons_frame = QFrame()
        buttons_frame.setLayout(buttons_layout)
        buttons_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f8f8;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 12px;
            }
        """)

        # ---------- CANVAS AREA ----------
        # canvases_layout = QHBoxLayout()
        # canvases_layout.addStretch(1)
        # canvases_layout.addWidget(self.left)
        # canvases_layout.addSpacing(GAP)
        # canvases_layout.addWidget(self.right)
        # canvases_layout.addStretch(1)

        canvases_layout = QHBoxLayout()
        canvases_layout.setSpacing(GAP)
        canvases_layout.setContentsMargins(0, 0, 0, 0)
        canvases_layout.addWidget(self.left)
        canvases_layout.addWidget(self.right)

        canvases_frame = QFrame()
        canvases_frame.setLayout(canvases_layout)
        # canvases_frame.setStyleSheet("""
        #     QFrame {
        #         background-color: #fafafa;
        #         border: 1px solid #ddd;
        #         border-radius: 8px;
        #         padding: 10px;
        #     }
        # """)

        # ---------- COUNT CLASS ----------
        # Simpan tombol agar mudah di-update nanti (dinamis sesuai model)
        self.class_buttons = {}

        self.class_counter_layout = QHBoxLayout()
        self.class_counter_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.class_counter_layout.setSpacing(8)
        self.class_counter_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder tombol contoh (ukuran tetap supaya tidak bergeser)
        for name in ["BAS", "EOS", "NEU", "LIM", "MON"]:
            btn = QPushButton(f"{name}: 0")
            btn.setEnabled(False)
            btn.setFixedWidth(100)   # ukuran seragam biar tidak geser
            btn.setFixedHeight(32)
            # btn.setStyleSheet("""
            #     QPushButton {
            #         background-color: #ffffff;
            #         border: 1px solid #bdbdbd;
            #         border-radius: 6px;
            #         font-weight: bold;
            #         color: #222;
            #     }
            # """)
            self.class_counter_layout.addWidget(btn)
            self.class_buttons[name] = btn

        self.class_counter_frame = QFrame()
        self.class_counter_frame.setLayout(self.class_counter_layout)
        # kurangi padding/frame supaya lebar mengikuti canvas lebih rapat
        self.class_counter_frame.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: none;
                padding: 4px;
            }
        """)
        # ---------- END COUNT CLASS ----------

        # ---------- PROGRESS BAR ----------
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                height: 14px;
            }
        """)

        # ---------- CENTRAL LAYOUT (div style) ----------
        # central_layout = QVBoxLayout()
        # central_layout.setSpacing(12)        # jarak antar “div”
        # central_layout.setContentsMargins(15, 15, 15, 15)

        # central_layout.addWidget(self.banner)
        # central_layout.addSpacing(8)
        # central_layout.addWidget(buttons_frame)
        # central_layout.addSpacing(10)
        # central_layout.addWidget(canvases_frame)
        # central_layout.addSpacing(10)
        # central_layout.addWidget(self.progress)

        # central = QWidget()
        # central.setLayout(central_layout)
        # self.setCentralWidget(central)

        # ---------- CENTRAL LAYOUT (div style) ----------
        central_layout = QVBoxLayout()
        central_layout.setSpacing(12)        # jarak antar “div”
        # kurangi margin kiri/kanan supaya canvases pas sesuai WINDOW_WIDTH
        central_layout.setContentsMargins(8, 12, 8, 12)

        central_layout.addWidget(self.banner)
        central_layout.addSpacing(8)
        central_layout.addWidget(buttons_frame)
        central_layout.addSpacing(10)
        central_layout.addWidget(canvases_frame)
        central_layout.addSpacing(6)
        central_layout.addWidget(self.class_counter_frame)
        central_layout.addSpacing(10)
        central_layout.addWidget(self.progress)

        central = QWidget()
        central.setLayout(central_layout)
        self.setCentralWidget(central)

        # ---------- STATUS BAR ----------
        # self.status = QStatusBar()
        # self.setStatusBar(self.status)

        # ---------- STATUS BAR ----------
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Buat label khusus untuk teks status di tengah
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                color: #333;
            }
        """)

        # Tambahkan label ke tengah status bar
        self.status.addPermanentWidget(self.status_label, 1)

        # ---------- CONNECT BUTTONS ----------
        self.btn_load.clicked.connect(self.load_model)
        self.btn_open_media.clicked.connect(self.open_media)
        self.btn_open_cam.clicked.connect(self.open_camera_dialog)
        self.btn_detect_folder.clicked.connect(self.detect_folder)
        self.btn_reset.clicked.connect(self.reset_views)
        self.btn_stop.clicked.connect(self.stop_worker)

        # Layout adaptif
        central.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.adjustSize()
        self.setFixedHeight(self.height())

        # ---------- WARNA LATAR MAIN WINDOW ----------
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f8f8;    /* abu lembut, tidak terlalu terang */
            }
        """)

        # self.btn_load.setFixedWidth(BUTTON_WIDTH)
        # self.btn_load.setFixedHeight(BUTTON_HEIGHT)

   # ---------------- END MAIN WINDOW ----------------

   # ---------------- Logika Button Switch ----------------

    def on_clahe_button_toggled(self):
        """
        Toggle CLAHE ON/OFF via button click.
        """
        is_on = self.btn_clahe.isChecked()
        if is_on:
            self.btn_clahe.setText("CLAHE: ON")
            # self.btn_clahe.setStyleSheet(
            #     "background-color: lightgreen; font-weight: bold; padding: 6px;")
        else:
            self.btn_clahe.setText("CLAHE: OFF")
            # self.btn_clahe.setStyleSheet(
            #     "background-color: #ccc; font-weight: bold; padding: 6px;")

        # update worker if running
        if self.worker:
            try:
                self.worker.use_clahe = is_on
            except Exception:
                pass

        # update status bar
        status = "CLAHE ON" if is_on else "CLAHE OFF"
        self.status_label.setText(f"{status} mode aktif")

    # ---------------- END Logika Button Switch ----------------

    # ---------------- Handlers ----------------

    # def load_model(self):
    #     path, _ = QFileDialog.getOpenFileName(
    #         self, "Pilih Model YOLOv8 (.pt/.onnx)", "", "Model Files (*.pt *.onnx)")
    #     if not path:
    #         return
    #     try:
    #         self.model = YOLO(path)
    #         # show CLAHE status as part of message (optional)
    #         clahe_status = " | CLAHE ON" if self.btn_clahe.isChecked() else ""
    #         self.status_label.setText(
    #             f"Model dimuat: {os.path.basename(path)} ({self.device}){clahe_status}")
    #         for b in [self.btn_open_media, self.btn_open_cam, self.btn_detect_folder, self.btn_reset, self.btn_stop]:
    #             b.setEnabled(True)
    #     except Exception as e:
    #         QMessageBox.critical(self, "Gagal muat model", str(e))
    #         self.model = None

    # -------------------------------------------------------------

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Model YOLOv8 (.pt/.onnx)", "", "Model Files (*.pt *.onnx)")
        if not path:
            return
        try:
            self.model = YOLO(path)
            clahe_status = " | CLAHE ON" if self.btn_clahe.isChecked() else ""

            # Enable tombol setelah model dimuat
            for b in [self.btn_open_media, self.btn_open_cam, self.btn_detect_folder, self.btn_reset, self.btn_stop]:
                b.setEnabled(True)

            # ----------- Rebuild class counter -----------
            # Hapus tombol lama
            for btn in self.class_buttons.values():
                btn.deleteLater()
            self.class_buttons.clear()

            # Ambil nama-nama class dari model
            try:
                class_names = list(self.model.names.values())
            except Exception:
                class_names = []

            # Buat ulang tombol class
            for name in class_names:
                btn = QPushButton(f"{name}: 0")
                btn.setEnabled(False)
                btn.setFixedWidth(100)
                btn.setFixedHeight(32)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #ffffff;
                        
                        color: #222;
                    }
                """)
                self.class_counter_layout.addWidget(btn)
                self.class_buttons[name] = btn

            self.status_label.setText(
                f"Model dimuat: {os.path.basename(path)} ({self.device}){clahe_status} | {len(class_names)} kelas")

        except Exception as e:
            QMessageBox.critical(self, "Gagal muat model", str(e))
            self.model = None

    # --------------------------------------------------------------
    def open_media(self):
        if self.model is None:
            QMessageBox.warning(self, "Peringatan",
                                "Muat model terlebih dahulu.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Media (Image/Video)", "", "Media (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        _, ext = os.path.splitext(path.lower())
        if ext in ('.mp4', '.avi', '.mov', '.mkv'):
            # video
            self._stop_worker()
            self.worker = InferenceWorker(self.model, device=self.device,
                                          use_clahe=self.btn_clahe.isChecked(),
                                          clahe_clip=self.clahe_clip, clahe_tile=self.clahe_tile)
            self.worker.configure_video(path)
            self._connect_worker()
            self.worker.start()
            self.progress.setVisible(False)
            self.status_label.setText("Playing video and detecting...")
        else:
            # image
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Error", "Gagal membaca gambar.")
                return
            frame_sq = center_crop_square(img, SQUARE)
            orig_rgb = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
            self.left.set_image_rgb(orig_rgb, reset=True)
            self.right.set_image_rgb(orig_rgb, reset=True)
            self._stop_worker()
            self.worker = InferenceWorker(self.model, device=self.device,
                                          use_clahe=self.btn_clahe.isChecked(),
                                          clahe_clip=self.clahe_clip, clahe_tile=self.clahe_tile)
            self.worker.configure_image(path)
            self._connect_worker()
            self.worker.start()
            self.status_label.setText(
                "Image loaded. Use pan/zoom to re-run detection.")
            # ensure stop button enabled
            self.btn_stop.setEnabled(True)

    def detect_available_cameras(self, max_tests=6):
        cams = []
        for i in range(max_tests):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cams.append(i)
                cap.release()
        return cams

    def open_camera_dialog(self):
        if self.model is None:
            QMessageBox.warning(self, "Peringatan",
                                "Muat model terlebih dahulu.")
            return
        cams = self.detect_available_cameras()
        if not cams:
            QMessageBox.warning(self, "No Camera",
                                "Tidak ada kamera terdeteksi.")
            return
        items = [f"Camera {i}" for i in cams]
        item, ok = QInputDialog.getItem(
            self, "Pilih Kamera", "Kamera:", items, 0, False)
        if not ok or not item:
            return
        cam_index = int(item.split()[-1])
        # try camera probe
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            QMessageBox.warning(
                self, "Error", f"Gagal buka kamera index {cam_index}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_H)
        time.sleep(0.12)
        cap.release()
        # start worker camera mode
        self._stop_worker()
        self.worker = InferenceWorker(self.model, device=self.device,
                                      use_clahe=self.btn_clahe.isChecked(),
                                      clahe_clip=self.clahe_clip, clahe_tile=self.clahe_tile)
        self.worker.configure_camera(cam_index)
        self._connect_worker()
        self.worker.start()
        self.status_label.setText(
            f"Camera {cam_index} opened (center-crop {SQUARE}x{SQUARE})")

    def detect_folder(self):
        if self.model is None:
            QMessageBox.warning(self, "Peringatan",
                                "Muat model terlebih dahulu.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Pilih Folder Gambar")
        if not folder:
            return
        # output base as requested: ../OutputTesting/<timestamp>_<foldername>
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.abspath(os.path.join(script_dir, '..'))
        base_out = os.path.join(
            parent, 'OutputTesting/OutputDataTestingDataset')
        os.makedirs(base_out, exist_ok=True)
        folder_name = os.path.basename(os.path.normpath(folder))
        ts = datetime.now().strftime('%y%m%d_%H%M%S')
        out_folder = os.path.join(base_out, f"{ts}_{folder_name}")
        os.makedirs(out_folder, exist_ok=True)

        self._stop_worker()
        self.worker = InferenceWorker(self.model, device=self.device,
                                      use_clahe=self.btn_clahe.isChecked(),
                                      clahe_clip=self.clahe_clip, clahe_tile=self.clahe_tile)
        self.worker.configure_folder(folder, out_folder)
        self._connect_worker()
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.worker.start()
        self.status_label.setText("Memproses folder...")

    def reset_views(self):
        self.left._scale = 1.0
        self.left._offset = [0.0, 0.0]
        self.left._update_display()
        self.left.emit_view_changed()
        self.right._scale = 1.0
        self.right._offset = [0.0, 0.0]
        self.right._update_display()

    def stop_worker(self):
        self._stop_worker()
        self.status_label.setText("Stopped.")

    def _stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1200)
        self.worker = None
        self.progress.setVisible(False)
        self.btn_stop.setEnabled(False)

    def _connect_worker(self):
        if not self.worker:
            return
        self.worker.frame_processed.connect(self.on_frame_processed)
        self.worker.folder_progress.connect(self.on_folder_progress)
        self.worker.folder_finished.connect(self.on_folder_finished)
        self.worker.error.connect(self.on_worker_error)
        self.btn_stop.setEnabled(True)

    # ---------------- Callbacks ----------------
    @Slot(int, int, int, int)
    def on_view_changed(self, x1, y1, x2, y2):
        # propagate view selection to worker
        if self.worker:
            self.worker.set_crop_rect((x1, y1, x2, y2))
            # if image mode, trigger new inference; for video/camera detection continuous will use rect automatically
            if self.worker._mode == 'image':
                self.worker.trigger_inference()
        # mirror view to right for alignment (not necessary but keeps sync)
        self.right.set_view(
            self.left._scale, self.left._offset[0], self.left._offset[1])

    @Slot(np.ndarray, np.ndarray, dict, float)
    # def on_frame_processed(self, orig_rgb, annotated_rgb, counts, dt):
    #     # orig_rgb and annotated_rgb are RGB arrays (SQUARE x SQUARE)
    #     self.left.set_image_rgb(orig_rgb, reset=False)
    #     self.right.set_image_rgb(annotated_rgb, reset=False)
    #     if counts:
    #         parts = [f"{v} {k}" for k, v in counts.items()]
    #         self.status_label.setText(f"Deteksi: {', '.join(parts)} | {dt:.3f}s")
    #     else:
    #         self.status_label.setText(f"Deteksi: 0 objek | {dt:.3f}s")
    @Slot(np.ndarray, np.ndarray, dict, float)
    def on_frame_processed(self, orig_rgb, annotated_rgb, counts, dt):
        self.left.set_image_rgb(orig_rgb, reset=False)
        self.right.set_image_rgb(annotated_rgb, reset=False)

        # Update counter class dinamis
        if hasattr(self, "class_buttons"):
            for name, btn in self.class_buttons.items():
                val = counts.get(name, 0)
                btn.setText(f"{name}: {val}")

                # Highlight jika ada deteksi
                if val > 0:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #d0ffd0;
                            
                            
                            color: #000;
                        }
                    """)
                else:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #ffffff;
                            
                            color: #222;
                        }
                    """)

        # Sembunyikan waktu dari status bar
        self.status.clearMessage()

    @Slot(int, int)
    def on_folder_progress(self, processed, total):
        if total:
            self.progress.setValue(int(processed / total * 100))
        self.status_label.setText(f"Memproses folder... {processed}/{total}")

    @Slot(str, float, float)
    def on_folder_finished(self, csv_path, total_time, avg):
        self.progress.setValue(100)
        self.progress.setVisible(False)
        QMessageBox.information(
            self, "Selesai", f"Folder selesai diproses.\nCSV: {csv_path}\nTotal {total_time:.2f}s | Avg {avg:.3f}s/img")
        self.status_label.setText("Selesai ✔")

    @Slot(str)
    def on_worker_error(self, msg):
        QMessageBox.critical(self, "Worker error", msg)
        self.status_label.setText("Error: " + msg)
        self.progress.setVisible(False)

    def on_clahe_toggled(self, state):
        """
        Update current worker's CLAHE flag dynamically when checkbox changes.
        """
        is_on = bool(state)
        if self.worker:
            try:
                self.worker.use_clahe = is_on
            except Exception:
                pass
        # update status line if model already loaded
        if self.model:
            clahe_status = "CLAHE ON" if is_on else "CLAHE OFF"
            self.status_label.setText(
                f"Model: {os.path.basename(self.model.path) if hasattr(self.model, 'path') else ''} ({self.device}) | {clahe_status}")

# --------------- main ---------------


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
