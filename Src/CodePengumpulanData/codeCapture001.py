#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera capture app (PySide6 + OpenCV)
- Pilih kamera (dropdown)
- FPS otomatis dari kamera
- 720x720 center crop preview
- Capture foto (.jpg) & video (.mp4)
- Filenames: yymmdd_HHMM
"""
import sys
import os
import cv2
from datetime import datetime
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QComboBox, QLineEdit
)


# ==============================================================
# Camera Thread
# ==============================================================
class CameraThread(QThread):
    frame_ready = Signal(object)
    fps_detected = Signal(float)

    def __init__(self, cam_index=0, width=1280, height=720):
        super().__init__()
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.running = False
        self.recording = False
        self.video_writer = None
        self.video_fps = 20
        self.out_path = None

    def run(self):
        """Try opening camera using several backends (Windows/Linux safe)."""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        cap = None
        for backend in backends:
            test = cv2.VideoCapture(self.cam_index, backend)
            if test.isOpened():
                cap = test
                print(
                    f"✅ Camera {self.cam_index} opened with backend {cap.getBackendName()}")
                break
            test.release()

        if cap is None or not cap.isOpened():
            print(f"❌ Gagal membuka kamera index {self.cam_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 20
        self.video_fps = fps
        self.fps_detected.emit(fps)

        self.running = True
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            self.frame_ready.emit(frame)
            if self.recording and self.video_writer is not None:
                self.video_writer.write(self._crop720(frame))
            self.msleep(5)

        cap.release()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def stop(self):
        self.running = False
        self.wait(1000)

    def _crop720(self, frame):
        h, w = frame.shape[:2]
        t = 720
        if w >= t and h >= t:
            x, y = (w - t)//2, (h - t)//2
            crop = frame[y:y+t, x:x+t]
            if crop.shape[:2] != (t, t):
                crop = cv2.resize(crop, (t, t))
            return crop
        return cv2.resize(frame, (t, t))

    def start_recording(self, path):
        size = (720, 720)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            path, fourcc, float(self.video_fps), size)
        if not self.video_writer.isOpened():
            self.video_writer = None
            raise RuntimeError("Gagal membuka VideoWriter.")
        self.recording = True

    def stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None


# ==============================================================
# Main Window
# ==============================================================
class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Capture - PySide6 (Manual Camera Select)")
        self.output_dir = os.path.join(os.getcwd(), "captures")
        os.makedirs(self.output_dir, exist_ok=True)

        # --- UI components ---
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(720, 720)
        self.preview_label.setStyleSheet(
            "background:#111; border:2px solid #444;")
        self.preview_label.setAlignment(Qt.AlignCenter)

        self.combo_camera = QComboBox()
        self.combo_camera.addItem("Select Camera...")
        self.btn_refresh = QPushButton("🔄")
        self.btn_photo = QPushButton("📸 Take Photo")
        self.btn_record = QPushButton("⏺ Start Recording")
        self.btn_stop = QPushButton("⏹ Stop Recording")
        self.btn_choose = QPushButton("📁 Folder")

        self.fps_label = QLineEdit("FPS: --")
        self.fps_label.setReadOnly(True)
        self.folder_line = QLineEdit(self.output_dir)
        self.folder_line.setReadOnly(True)

        # Layout top controls
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(self.combo_camera)
        cam_layout.addWidget(self.btn_refresh)
        cam_layout.addWidget(self.fps_label)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(self.btn_photo)
        ctrl_layout.addWidget(self.btn_record)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_choose)

        layout = QVBoxLayout(self)
        layout.addLayout(cam_layout)
        layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)
        layout.addLayout(ctrl_layout)
        layout.addWidget(self.folder_line)

        # Init thread
        self.thread = None
        self.current_frame = None
        self.is_recording = False

        # Signals
        self.btn_photo.clicked.connect(self.take_photo)
        self.btn_record.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        self.btn_choose.clicked.connect(self.choose_folder)
        self.btn_refresh.clicked.connect(self.refresh_cameras)
        self.combo_camera.currentIndexChanged.connect(self.change_camera)

        self.btn_stop.setEnabled(False)

        # initial populate
        self.refresh_cameras()

    # ==============================================================
    # Camera management
    # ==============================================================
    def refresh_cameras(self):
        current = self.combo_camera.currentText()
        self.combo_camera.clear()
        self.combo_camera.addItem("Select Camera...")
        available = self._list_cameras()
        if not available:
            self.combo_camera.addItem("No camera found")
            return
        for idx, name in available:
            self.combo_camera.addItem(f"{idx}: {name}", idx)
        print(f"📷 Detected {len(available)} camera(s).")

    def _list_cameras(self, max_test=10):
        cams = []
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        print("🔍 Scanning camera devices...")
        for i in range(max_test):
            found = False
            for backend in backends:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        name = f"Camera {i} ({cap.getBackendName()})"
                        cams.append((i, name))
                        print(
                            f"✅ Found camera index {i} using backend {cap.getBackendName()}")
                        found = True
                    cap.release()
                    if found:
                        break
                else:
                    cap.release()
            if not found:
                print(f"❌ No camera at index {i}")
        return cams

    def change_camera(self, index):
        data = self.combo_camera.currentData()
        if data is None:
            return  # skip "Select Camera..." or "No camera found"
        cam_index = int(data)

        # stop any running thread first
        if self.thread:
            self.thread.stop()
            self.thread = None

        print(f"🎥 Starting camera index {cam_index}")
        self.thread = CameraThread(cam_index)
        self.thread.frame_ready.connect(self.on_frame)
        self.thread.fps_detected.connect(self.on_fps_detected)
        self.thread.start()

    # ==============================================================
    # Frame update
    # ==============================================================
    @Slot(object)
    def on_frame(self, frame):
        self.current_frame = frame
        disp = self._crop_720(frame)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    @Slot(float)
    def on_fps_detected(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")
        print(f"🎞 FPS kamera aktif: {fps:.1f}")

    def _crop_720(self, frame):
        h, w = frame.shape[:2]
        t = 720
        if w >= t and h >= t:
            x, y = (w - t)//2, (h - t)//2
            crop = frame[y:y+t, x:x+t]
            if crop.shape[:2] != (t, t):
                crop = cv2.resize(crop, (t, t))
            return crop
        return cv2.resize(frame, (t, t))

    # ==============================================================
    # Actions
    # ==============================================================
    def _timestamp(self):
        return datetime.now().strftime("%y%m%d_%H%M_%S")

    def take_photo(self):
        if self.current_frame is None:
            QMessageBox.warning(
                self, "No Frame", "Belum ada frame dari kamera.")
            return
        path = os.path.join(self.output_dir, f"{self._timestamp()}.jpg")
        cv2.imwrite(path, self._crop_720(self.current_frame))
        QMessageBox.information(self, "Saved", f"Photo saved:\n{path}")

    def start_recording(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Frame", "Belum ada frame.")
            return
        if not self.thread:
            QMessageBox.warning(self, "Error", "Thread kamera tidak aktif.")
            return
        path = os.path.join(self.output_dir, f"{self._timestamp()}.mp4")
        try:
            self.thread.start_recording(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        self.is_recording = True
        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)
        QMessageBox.information(self, "Recording", f"Recording to:\n{path}")

    def stop_recording(self):
        if not self.is_recording:
            return
        self.thread.stop_recording()
        self.is_recording = False
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.information(self, "Stopped", "Recording stopped.")

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose folder", self.output_dir)
        if folder:
            self.output_dir = folder
            self.folder_line.setText(folder)

    def closeEvent(self, e):
        if self.thread:
            if self.is_recording:
                self.thread.stop_recording()
            self.thread.stop()
        e.accept()


# ==============================================================
# Run App
# ==============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CameraApp()
    w.setFixedSize(820, 880)
    w.show()
    sys.exit(app.exec())
