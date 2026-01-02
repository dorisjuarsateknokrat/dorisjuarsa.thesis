#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Microscope GUI - Realtime Preview + REC Toggle (Thesis-ready)
- Buttons kept: Load Model, Open Media, Open Camera, Reset Position, Stop, CLAHE
- Added: REC toggle (like CLAHE)
- Camera mode:
  - Open Camera => preview + detection (no recording)
  - REC ON => start session recording (CSV+MP4)
  - REC OFF => stop recording (camera preview continues)
  - Stop => stop camera worker completely
- Output root:
  Data/DataTesting/Output/Realtime/<session_folder>/
- Session folder:
  <YYYYMMDD_HHMMSS>_cam<idx>_<modelstem>_claheON|OFF
- Cross-OS paths via pathlib

MOD (requested):
- Watermark ONLY on exported video (session.mp4):
  - class counts + total
  - inference_ms and fps_inst (same values written to CSV)
"""

from __future__ import annotations

import sys
import time
import csv
from datetime import datetime
from pathlib import Path
import threading
import re

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QMessageBox, QHBoxLayout, QVBoxLayout, QProgressBar, QStatusBar,
    QInputDialog, QSizePolicy, QFrame
)

# ---------------- CONFIG ----------------
SQUARE = 720
GAP = 5
WINDOW_WIDTH = SQUARE * 2 + GAP + 40

DEFAULT_CAMERA_W = 1280
DEFAULT_CAMERA_H = 720

# <-- sesuai permintaan (lebih ketat, kurangi false positive)
DEFAULT_CONF = 0.5
DEFAULT_IMGSZ = 640      # konsisten dengan training imgsz=640 (boleh tetap)

# Untuk video recording: estimasi fps dari beberapa frame pertama saat REC mulai
BUFFER_FRAMES_FOR_FPS = 30
MIN_WRITER_FPS = 5.0
MAX_WRITER_FPS = 60.0

BUTTON_WIDTH = 190
BUTTON_HEIGHT = 30


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


def safe_slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s.strip("_")


def ts_ms() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ---------------- ImageLabel ----------------
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
        scaled = self._pixmap.scaled(
            int(orig_w * self._scale), int(orig_h * self._scale),
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
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
        old_scale = self._scale
        self._scale = max(0.2, min(self._scale * factor, 6.0))

        pos = event.position()
        mx, my = pos.x(), pos.y()
        old_img_x = (mx - self._offset[0]) / old_scale
        old_img_y = (my - self._offset[1]) / old_scale
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


# ---------------- Worker Thread ----------------
class InferenceWorker(QThread):
    # orig_rgb, annotated_rgb, counts, dt_sec
    frame_processed = Signal(np.ndarray, np.ndarray, dict, float)
    # session_dir
    session_started = Signal(str)
    # session_dir
    session_stopped = Signal(str)
    error = Signal(str)

    def __init__(self, model: YOLO, device: str, use_clahe: bool, clahe_clip: float, clahe_tile: int,
                 project_root: Path):
        super().__init__()
        self.model = model
        self.device = device
        self._running = False

        self._mode = None
        self._input_path: str | None = None
        self._cam_index: int | None = None
        self._crop_rect = None
        self._trigger = threading.Event()

        # CLAHE
        self.use_clahe = bool(use_clahe)
        self.clahe_clip = float(clahe_clip)
        self.clahe_tile = int(clahe_tile)

        # Project paths
        self.project_root = project_root

        # Session/logging state
        self._recording = False
        self._recording_lock = threading.Lock()
        # None=no change, True=start, False=stop
        self._recording_request: bool | None = None

        self.session_dir: Path | None = None
        self.csv_path: Path | None = None
        self.video_path: Path | None = None
        self._csv_file = None
        self._csv_writer = None
        self._writer = None

        # FPS estimation buffer (only when recording starts)
        self._buffer_frames = []
        self._buffer_start_time = None

        # class names (same as GUI counter)
        try:
            self.class_names = list(self.model.names.values())
        except Exception:
            self.class_names = []

        self._frame_idx = 0  # frame index within recording session only
        self._t_session_start = None

    # -------- configuration --------
    def configure_camera(self, index: int):
        self._mode = "camera"
        self._cam_index = index

    def configure_image(self, path: str):
        self._mode = "image"
        self._input_path = path

    def configure_video(self, path: str):
        self._mode = "video"
        self._input_path = path

    def set_crop_rect(self, rect):
        self._crop_rect = rect

    def trigger_inference(self):
        self._trigger.set()

    # -------- recording control (called from GUI thread) --------
    def request_recording(self, enable: bool):
        with self._recording_lock:
            self._recording_request = bool(enable)

    def is_recording(self) -> bool:
        with self._recording_lock:
            return bool(self._recording)

    def stop(self):
        self._running = False
        self._trigger.set()

    # -------- internals --------
    def _apply_clahe(self, bgr_img):
        try:
            lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=(self.clahe_tile, self.clahe_tile)
            )
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            return bgr_img

    def _infer(self, bgr_img):
        t0 = time.time()
        results = self.model(
            bgr_img,
            device=self.device,
            verbose=False,
            conf=DEFAULT_CONF,
            imgsz=DEFAULT_IMGSZ
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        return results, dt

    def _draw_boxes(self, base_bgr, results):
        ann = base_bgr.copy()
        res = results[0].to("cpu")
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return ann
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = res.names
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            label = names.get(int(cls), str(cls))
            col = label_color(label)
            cv2.rectangle(ann, (x1i, y1i), (x2i, y2i), col, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            by = max(0, y1i - th - 6)
            cv2.rectangle(ann, (x1i, by), (x1i + tw + 6, by + th + 4), col, -1)
            cv2.putText(ann, text, (x1i + 3, by + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return ann

    def _counts_from_results(self, results) -> dict:
        res = results[0].to("cpu")
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {}
        labels = boxes.cls.cpu().numpy().astype(int)
        names = res.names
        counts = {}
        for l in labels:
            n = names.get(int(l), str(l))
            counts[n] = counts.get(n, 0) + 1
        return counts

    # ===== MOD: watermark only for export video =====
    def _overlay_watermark_for_export(self, frame_bgr: np.ndarray, counts: dict,
                                      inference_ms: float, fps_inst: float) -> np.ndarray:
        """
        Watermark ONLY untuk video export (session.mp4):
        - Menampilkan inference_ms & fps_inst
        - Menampilkan count tiap class + total
        Tidak mengubah preview UI (karena hanya dipakai untuk writer).
        """
        if frame_bgr is None:
            return frame_bgr

        out = frame_bgr.copy()
        h, w = out.shape[:2]

        line1 = f"inference_ms={inference_ms:.3f} | fps_inst={fps_inst:.3f}"

        total_objects = 0
        parts = []
        for cn in self.class_names:
            v = int(counts.get(cn, 0))
            total_objects += v
            parts.append(f"{cn}={v}")
        parts.append(f"total={total_objects}")

        max_per_line = 4
        count_lines = []
        for i in range(0, len(parts), max_per_line):
            count_lines.append(" | ".join(parts[i:i + max_per_line]))

        lines = [line1] + count_lines

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        pad = 8
        line_gap = 6

        text_sizes = [cv2.getTextSize(t, font, font_scale, thickness)[
            0] for t in lines]
        block_w = max(ts[0] for ts in text_sizes) + pad * 2
        block_h = sum(ts[1] for ts in text_sizes) + \
            pad * 2 + line_gap * (len(lines) - 1)

        x0 = 10
        y0 = h - 10 - block_h
        x0 = max(0, min(x0, w - block_w))
        y0 = max(0, min(y0, h - block_h))

        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + block_w,
                      y0 + block_h), (0, 0, 0), -1)
        alpha = 0.45
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

        y = y0 + pad
        for t, (tw, th) in zip(lines, text_sizes):
            y_text = y + th
            cv2.putText(out, t, (x0 + pad, y_text), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            y = y_text + line_gap

        return out

    def _make_session_dir(self) -> Path:
        # Root: Data/DataTesting/Output/Realtime
        out_root = (self.project_root / "Data" / "DataTesting" /
                    "Output" / "Realtime").resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        ts_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        clahe_tag = "claheON" if self.use_clahe else "claheOFF"

        model_stem = "model"
        try:
            p = getattr(self.model, "ckpt_path", None)
            if p:
                model_stem = Path(p).stem
        except Exception:
            pass

        session_folder = f"{ts_folder}_cam{self._cam_index}_{safe_slug(model_stem)}_{clahe_tag}"
        session_dir = (out_root / session_folder).resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _start_recording(self):
        # already recording?
        if self._recording:
            return

        self.session_dir = self._make_session_dir()
        self.csv_path = self.session_dir / "session.csv"
        self.video_path = self.session_dir / "session.mp4"

        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        header = ["frame_idx", "timestamp",
                  "inference_ms", "fps_inst", "clahe_on"]
        for cn in self.class_names:
            header.append(f"count_{cn}")
        header.append("total_objects")
        self._csv_writer.writerow(header)

        # writer: init after buffering for fps estimation
        self._buffer_frames = []
        self._buffer_start_time = time.time()
        self._writer = None

        self._frame_idx = 0
        self._t_session_start = time.time()

        self._recording = True
        self.session_started.emit(str(self.session_dir))

    def _init_writer_if_ready(self):
        if self._writer is not None:
            return
        if len(self._buffer_frames) < BUFFER_FRAMES_FOR_FPS:
            return

        elapsed = max(1e-6, time.time() -
                      (self._buffer_start_time or time.time()))
        fps_est = len(self._buffer_frames) / elapsed
        fps_est = float(np.clip(fps_est, MIN_WRITER_FPS, MAX_WRITER_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.video_path), fourcc, fps_est, (SQUARE, SQUARE))

        for fr in self._buffer_frames:
            self._writer.write(fr)
        self._buffer_frames.clear()

    def _stop_recording(self):
        if not self._recording:
            return

        # if writer not yet created, create with fallback fps and flush buffer
        try:
            if self._writer is None:
                elapsed = max(1e-6, time.time() -
                              (self._buffer_start_time or time.time()))
                fps_fallback = len(self._buffer_frames) / \
                    elapsed if self._buffer_frames else 10.0
                fps_fallback = float(
                    np.clip(fps_fallback, MIN_WRITER_FPS, MAX_WRITER_FPS))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(
                    str(self.video_path), fourcc, fps_fallback, (SQUARE, SQUARE))
                for fr in self._buffer_frames:
                    self._writer.write(fr)
                self._buffer_frames.clear()
        except Exception:
            pass

        try:
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
        self._writer = None

        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass
        self._csv_file = None
        self._csv_writer = None

        finished_dir = str(self.session_dir) if self.session_dir else ""
        self.session_dir = None
        self.csv_path = None
        self.video_path = None

        self._recording = False
        self.session_stopped.emit(finished_dir)

    def _apply_recording_request_if_any(self):
        with self._recording_lock:
            req = self._recording_request
            self._recording_request = None

        if req is None:
            return
        if req is True and self._mode == "camera":
            self._start_recording()
        elif req is False:
            self._stop_recording()

    # -------- main run --------
    def run(self):
        if self.model is None:
            self.error.emit("Model belum dimuat.")
            return

        self._running = True
        try:
            if self._mode == "camera":
                cap = cv2.VideoCapture(
                    self._cam_index if self._cam_index is not None else 0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_H)
                if not cap.isOpened():
                    self.error.emit("Gagal membuka kamera.")
                    return

                # preview loop (recording OFF by default)
                while self._running:
                    self._apply_recording_request_if_any()

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

                    img_proc = self._apply_clahe(
                        img_for_infer) if self.use_clahe else img_for_infer
                    results, dt = self._infer(img_proc)
                    ann_bgr = self._draw_boxes(img_proc, results)

                    if offset_x != 0 or offset_y != 0:
                        full_annot_rgb = cv2.cvtColor(
                            frame_sq, cv2.COLOR_BGR2RGB)
                        ah, aw = ann_bgr.shape[:2]
                        full_annot_rgb[offset_y:offset_y+ah, offset_x:offset_x +
                                       aw] = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

                        ann_for_video_bgr = frame_sq.copy()
                        ann_for_video_bgr[offset_y:offset_y +
                                          ah, offset_x:offset_x+aw] = ann_bgr
                    else:
                        full_annot_rgb = cv2.cvtColor(
                            ann_bgr, cv2.COLOR_BGR2RGB)
                        ann_for_video_bgr = ann_bgr

                    counts = self._counts_from_results(results)
                    self.frame_processed.emit(
                        orig_rgb, full_annot_rgb, counts, dt)

                    # ---- write only if recording ON ----
                    if self._recording and self._csv_writer is not None:
                        self._frame_idx += 1
                        inference_ms = dt * 1000.0
                        fps_inst = (1.0 / dt) if dt > 0 else 0.0
                        clahe_on = 1 if self.use_clahe else 0

                        row = [
                            self._frame_idx,
                            ts_ms(),
                            round(inference_ms, 3),
                            round(fps_inst, 3),
                            clahe_on
                        ]
                        total_objects = 0
                        for cn in self.class_names:
                            v = int(counts.get(cn, 0))
                            row.append(v)
                            total_objects += v
                        row.append(total_objects)

                        self._csv_writer.writerow(row)

                        # ===== MOD: watermark ONLY on exported video frames =====
                        ann_for_video_bgr_wm = self._overlay_watermark_for_export(
                            ann_for_video_bgr, counts, inference_ms, fps_inst
                        )

                        if self._writer is None:
                            self._buffer_frames.append(ann_for_video_bgr_wm)
                            self._init_writer_if_ready()
                        else:
                            self._writer.write(ann_for_video_bgr_wm)

                    QThread.msleep(1)

                cap.release()

                # if user closes app / stop worker while recording: close session gracefully
                if self._recording:
                    self._stop_recording()

            elif self._mode == "video":
                cap = cv2.VideoCapture(str(self._input_path))
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

                    img_proc = self._apply_clahe(
                        img_for_infer) if self.use_clahe else img_for_infer
                    results, dt = self._infer(img_proc)
                    ann_bgr = self._draw_boxes(img_proc, results)

                    if offset_x != 0 or offset_y != 0:
                        full_annot_rgb = cv2.cvtColor(
                            frame_sq, cv2.COLOR_BGR2RGB)
                        ah, aw = ann_bgr.shape[:2]
                        full_annot_rgb[offset_y:offset_y+ah, offset_x:offset_x +
                                       aw] = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        full_annot_rgb = cv2.cvtColor(
                            ann_bgr, cv2.COLOR_BGR2RGB)

                    counts = self._counts_from_results(results)
                    self.frame_processed.emit(
                        orig_rgb, full_annot_rgb, counts, dt)
                    QThread.msleep(1)
                cap.release()

            elif self._mode == "image":
                img = cv2.imread(str(self._input_path))
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

                img_proc = self._apply_clahe(
                    img_for_infer) if self.use_clahe else img_for_infer
                results, dt = self._infer(img_proc)
                ann_bgr = self._draw_boxes(img_proc, results)

                if offset_x != 0 or offset_y != 0:
                    full_annot_rgb = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
                    ah, aw = ann_bgr.shape[:2]
                    full_annot_rgb[offset_y:offset_y+ah, offset_x:offset_x +
                                   aw] = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                else:
                    full_annot_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

                counts = self._counts_from_results(results)
                self.frame_processed.emit(orig_rgb, full_annot_rgb, counts, dt)

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

                        img_proc = self._apply_clahe(
                            img_for_infer) if self.use_clahe else img_for_infer
                        results, dt = self._infer(img_proc)
                        ann_bgr = self._draw_boxes(img_proc, results)

                        if offset_x != 0 or offset_y != 0:
                            full_annot_rgb = cv2.cvtColor(
                                frame_sq, cv2.COLOR_BGR2RGB)
                            ah, aw = ann_bgr.shape[:2]
                            full_annot_rgb[offset_y:offset_y+ah, offset_x:offset_x +
                                           aw] = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                        else:
                            full_annot_rgb = cv2.cvtColor(
                                ann_bgr, cv2.COLOR_BGR2RGB)

                        counts = self._counts_from_results(results)
                        self.frame_processed.emit(
                            orig_rgb, full_annot_rgb, counts, dt)

        except Exception as e:
            self.error.emit(str(e))
            # try close recording if needed
            try:
                if self._recording:
                    self._stop_recording()
            except Exception:
                pass
        finally:
            self._running = False


# ---------------- Main Window ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DORIS JUARSA - 24321025")
        self.setFixedWidth(WINDOW_WIDTH)

        # Project root: .../DORISJUARSA.THESIS
        # Current file: Src/CodeTesting/codeTesting001.py -> root = parents[2]
        self.project_root = Path(__file__).resolve().parents[2]

        # State
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model: YOLO | None = None
        self.worker: InferenceWorker | None = None

        # CLAHE defaults
        self.clahe_clip = 2.0
        self.clahe_tile = 8

        # UI canvases
        self.left = ImageLabel()
        self.right = ImageLabel()
        self.left.view_changed.connect(self.on_view_changed)

        # Banner
        banner_text = """
        <div style="text-align:center;">
            <div style="font-size:16pt;">Model Deteksi Objek Real-Time Untuk Klasifikasi Otomatis Subtipe Leukosit Pada Citra Mikroskopis Darah Dengan YOLOv8</div>
            <div style="font-size:11pt; margin-top:4px;">Dibuat oleh: Doris Juarsa || NPM: 24321025</div>
            <div style="font-size:11pt; margin-top:4px;">Pembimbing: Dr. Erliyan Redy Susanto, S.Kom., M.Kom. || Penguji 1: Dr. Rohmat Indra Borman, M.Kom.</div>
        </div>
        """
        self.banner = QLabel()
        self.banner.setText(banner_text)
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 4px;
        """)

        # Buttons
        self.btn_load = QPushButton("Load Model")
        self.btn_open_media = QPushButton("Open Media")
        self.btn_open_cam = QPushButton("Open Camera")
        self.btn_reset = QPushButton("Reset Position")
        self.btn_stop = QPushButton("Stop")

        self.btn_clahe = QPushButton("CLAHE: OFF")
        self.btn_clahe.setCheckable(True)
        self.btn_clahe.setChecked(False)
        self.btn_clahe.clicked.connect(self.on_clahe_button_toggled)

        # REC toggle (like CLAHE)
        self.btn_rec = QPushButton("REC: OFF")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setChecked(False)
        self.btn_rec.clicked.connect(self.on_rec_toggled)

        for b in [self.btn_load, self.btn_open_media, self.btn_open_cam, self.btn_reset, self.btn_stop, self.btn_clahe, self.btn_rec]:
            b.setFixedWidth(BUTTON_WIDTH)
            b.setFixedHeight(BUTTON_HEIGHT)

        # disable before model loaded
        for b in [self.btn_open_media, self.btn_open_cam, self.btn_reset, self.btn_stop, self.btn_clahe, self.btn_rec]:
            b.setEnabled(False)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        for b in [self.btn_load, self.btn_open_media, self.btn_open_cam, self.btn_reset, self.btn_stop, self.btn_clahe, self.btn_rec]:
            buttons_layout.addWidget(b)

        buttons_frame = QFrame()
        buttons_frame.setLayout(buttons_layout)
        buttons_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f8f8;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 2px;
            }
        """)

        canvases_layout = QHBoxLayout()
        canvases_layout.setSpacing(GAP)
        canvases_layout.setContentsMargins(0, 0, 0, 0)
        canvases_layout.addWidget(self.left)
        canvases_layout.addWidget(self.right)

        canvases_frame = QFrame()
        canvases_frame.setLayout(canvases_layout)

        # Class counter (dynamic)
        self.class_buttons: dict[str, QPushButton] = {}
        self.class_counter_layout = QHBoxLayout()
        self.class_counter_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.class_counter_layout.setSpacing(8)
        self.class_counter_layout.setContentsMargins(0, 0, 0, 0)

        self.class_counter_frame = QFrame()
        self.class_counter_frame.setLayout(self.class_counter_layout)
        self.class_counter_frame.setStyleSheet(
            "QFrame { background-color: transparent; border: none; padding: 4px; }")

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        central_layout = QVBoxLayout()
        central_layout.setSpacing(12)
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

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "QLabel { font-size: 11pt; color: #333; }")
        self.status.addPermanentWidget(self.status_label, 1)

        # Connect
        self.btn_load.clicked.connect(self.load_model)
        self.btn_open_media.clicked.connect(self.open_media)
        self.btn_open_cam.clicked.connect(self.open_camera_dialog)
        self.btn_reset.clicked.connect(self.reset_views)
        self.btn_stop.clicked.connect(self.stop_worker)

        self.adjustSize()
        self.setFixedHeight(self.height())
        self.setStyleSheet("QMainWindow { background-color: #f8f8f8; }")

    # -------- Helpers for UI state --------
    def _set_rec_ui(self, is_on: bool):
        self.btn_rec.blockSignals(True)
        self.btn_rec.setChecked(is_on)
        self.btn_rec.setText("REC: ON" if is_on else "REC: OFF")
        self.btn_rec.blockSignals(False)

    def _set_clahe_ui(self, is_on: bool):
        self.btn_clahe.blockSignals(True)
        self.btn_clahe.setChecked(is_on)
        self.btn_clahe.setText("CLAHE: ON" if is_on else "CLAHE: OFF")
        self.btn_clahe.blockSignals(False)

    # -------- CLAHE --------
    def on_clahe_button_toggled(self):
        is_on = self.btn_clahe.isChecked()
        self.btn_clahe.setText("CLAHE: ON" if is_on else "CLAHE: OFF")

        # If recording is ON, we keep it running, but CLAHE change will affect subsequent frames
        if self.worker:
            self.worker.use_clahe = is_on

        # status only
        self.status_label.setText("CLAHE ON" if is_on else "CLAHE OFF")

    # -------- REC toggle --------
    def on_rec_toggled(self):
        is_on = self.btn_rec.isChecked()

        # Only meaningful in camera mode with running worker
        if not self.worker or not self.worker.isRunning() or getattr(self.worker, "_mode", None) != "camera":
            # revert toggle
            self._set_rec_ui(False)
            QMessageBox.warning(
                self, "REC", "REC hanya bisa digunakan saat mode Camera aktif.")
            return

        # Apply recording request to worker
        self.worker.request_recording(is_on)

        # Update button label
        self.btn_rec.setText("REC: ON" if is_on else "REC: OFF")

        if is_on:
            self.status_label.setText("REC ● Recording started...")
        else:
            self.status_label.setText(
                "REC ■ Recording stopped (preview continues)")

    # -------- Model --------
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Model YOLOv8 (.pt/.onnx)", "", "Model Files (*.pt *.onnx)"
        )
        if not path:
            return
        try:
            self.model = YOLO(path)

            # enable buttons after model loaded
            for b in [self.btn_open_media, self.btn_open_cam, self.btn_reset, self.btn_stop, self.btn_clahe, self.btn_rec]:
                b.setEnabled(True)

            # rebuild class counter from model.names (same as your previous GUI)
            for btn in self.class_buttons.values():
                btn.deleteLater()
            self.class_buttons.clear()

            class_names = list(self.model.names.values())
            for name in class_names:
                btn = QPushButton(f"{name}: 0")
                btn.setEnabled(False)
                btn.setFixedWidth(130)
                btn.setFixedHeight(32)
                btn.setStyleSheet(
                    "QPushButton { background-color:#ffffff; color:#222; }")
                self.class_counter_layout.addWidget(btn)
                self.class_buttons[name] = btn

            clahe_status = "CLAHE ON" if self.btn_clahe.isChecked() else "CLAHE OFF"
            self.status_label.setText(
                f"Model dimuat: {Path(path).name} ({self.device}) | {clahe_status} | conf={DEFAULT_CONF}")

        except Exception as e:
            QMessageBox.critical(self, "Gagal muat model", str(e))
            self.model = None

    # -------- Media (optional in thesis) --------
    def open_media(self):
        if self.model is None:
            QMessageBox.warning(self, "Peringatan",
                                "Muat model terlebih dahulu.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Media (Image/Video)", "",
            "Media (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return

        ext = Path(path).suffix.lower()

        self._stop_worker_internal()

        self.worker = InferenceWorker(
            model=self.model,
            device=self.device,
            use_clahe=self.btn_clahe.isChecked(),
            clahe_clip=self.clahe_clip,
            clahe_tile=self.clahe_tile,
            project_root=self.project_root
        )

        if ext in ('.mp4', '.avi', '.mov', '.mkv'):
            self.worker.configure_video(path)
            self.status_label.setText(
                "Video mode: detecting... (Stop untuk berhenti)")
        else:
            self.worker.configure_image(path)
            self.status_label.setText(
                "Image mode: pan/zoom untuk re-run detection.")

        self._connect_worker()
        self.worker.start()
        self.btn_stop.setEnabled(True)

        # REC not applicable outside camera
        self._set_rec_ui(False)

    # -------- Camera --------
    def detect_available_cameras(self, max_tests=8):
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

        # Stop any previous worker
        self._stop_worker_internal()

        # Start preview worker (recording OFF by default)
        self.worker = InferenceWorker(
            model=self.model,
            device=self.device,
            use_clahe=self.btn_clahe.isChecked(),
            clahe_clip=self.clahe_clip,
            clahe_tile=self.clahe_tile,
            project_root=self.project_root
        )
        self.worker.configure_camera(cam_index)
        self._connect_worker()
        self.worker.start()

        self.btn_stop.setEnabled(True)

        # REC starts OFF
        self._set_rec_ui(False)

        self.status_label.setText(
            f"Camera {cam_index} preview + detect (tekan REC untuk mulai rekam)")

    # -------- Reset / Stop --------
    def reset_views(self):
        self.left._scale = 1.0
        self.left._offset = [0.0, 0.0]
        self.left._update_display()
        self.left.emit_view_changed()

        self.right._scale = 1.0
        self.right._offset = [0.0, 0.0]
        self.right._update_display()

    def stop_worker(self):
        # Stop entire worker (media/video/camera). If recording ON, worker will close session gracefully.
        self._stop_worker_internal()
        self.status_label.setText("Stopped.")
        self._set_rec_ui(False)

    def _stop_worker_internal(self):
        if self.worker and self.worker.isRunning():
            try:
                # ensure stop recording request is applied quickly
                self.worker.request_recording(False)
            except Exception:
                pass
            self.worker.stop()
            self.worker.wait(2500)
        self.worker = None
        self.btn_stop.setEnabled(False)

    # -------- Signals --------
    def _connect_worker(self):
        if not self.worker:
            return
        self.worker.frame_processed.connect(self.on_frame_processed)
        self.worker.session_started.connect(self.on_session_started)
        self.worker.session_stopped.connect(self.on_session_stopped)
        self.worker.error.connect(self.on_worker_error)

    @Slot(int, int, int, int)
    def on_view_changed(self, x1, y1, x2, y2):
        if self.worker:
            self.worker.set_crop_rect((x1, y1, x2, y2))
            if getattr(self.worker, "_mode", None) == "image":
                self.worker.trigger_inference()
        self.right.set_view(
            self.left._scale, self.left._offset[0], self.left._offset[1])

    @Slot(np.ndarray, np.ndarray, dict, float)
    def on_frame_processed(self, orig_rgb, annotated_rgb, counts, dt):
        self.left.set_image_rgb(orig_rgb, reset=False)
        self.right.set_image_rgb(annotated_rgb, reset=False)

        # update class counters
        for name, btn in self.class_buttons.items():
            val = int(counts.get(name, 0))
            btn.setText(f"{name}: {val}")
            if val > 0:
                btn.setStyleSheet(
                    "QPushButton { background-color:#d0ffd0; color:#000; }")
            else:
                btn.setStyleSheet(
                    "QPushButton { background-color:#ffffff; color:#222; }")

    @Slot(str)
    def on_session_started(self, session_dir):
        # keep REC ON in UI
        self._set_rec_ui(True)
        self.status_label.setText(f"REC ● Started: {session_dir}")

    @Slot(str)
    def on_session_stopped(self, session_dir):
        # keep REC OFF in UI
        self._set_rec_ui(False)
        if session_dir:
            QMessageBox.information(
                self, "Recording selesai", f"Output tersimpan di:\n{session_dir}")
            self.status_label.setText(
                "REC ■ Recording stopped (preview continues)")
        else:
            self.status_label.setText("REC ■ Recording stopped")

    @Slot(str)
    def on_worker_error(self, msg):
        QMessageBox.critical(self, "Worker error", msg)
        self.status_label.setText("Error: " + msg)
        # stop rec ui to safe state
        self._set_rec_ui(False)


# --------------- main ---------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
