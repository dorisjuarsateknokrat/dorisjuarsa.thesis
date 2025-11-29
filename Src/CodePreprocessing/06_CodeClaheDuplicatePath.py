from tkinter import ttk, messagebox
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageTk
from datetime import datetime
import numpy as np
import cv2
import shutil
import random
import glob
import os

# === KONFIGURASI ===
DATASET_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/DorisjuarsaDatasetBalance360x360RotateScale15_2509301304"

GRID_ROWS = 2
GRID_COLS = 2
GRID_SIZE = GRID_ROWS * GRID_COLS

MAX_PREVIEW_SIZE = (400, 400)

# Ambil gambar acak untuk preview
all_img_paths = []
for split in ["train", "val", "test"]:
    img_dir = os.path.join(DATASET_ROOT, "images", split)
    all_img_paths.extend(glob.glob(os.path.join(img_dir, "*.*")))

if len(all_img_paths) < GRID_SIZE:
    raise ValueError(f"Dataset minimal {GRID_SIZE} gambar untuk preview!")

preview_images = random.sample(all_img_paths, GRID_SIZE)


# === Fungsi CLAHE ===
def apply_clahe_to_image(img_path, clip_limit=2.0, tile_grid_size=8):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Tidak bisa baca gambar: {img_path}")

    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
        )
        result = clahe.apply(img)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def save_clahe_image(args):
    img_path, out_path, clip_limit, tile_grid_size = args
    pil_img = apply_clahe_to_image(img_path, clip_limit, tile_grid_size)
    pil_img.save(out_path)


# === GUI ===
class CLAHEPreviewApp:
    def __init__(self, root, image_paths):
        self.root = root
        self.image_paths = image_paths
        self.clip_limit = 2.0
        self.tile_grid_size = 8
        self.photo_images = []

        root.title("CLAHE Preview - Dorisjuarsa Tesis")

        # === MAXIMIZE WINDOW CROSS-PLATFORM ===
        try:
            root.state("zoomed")  # Windows, Linux (umumnya)
        except tk.TclError:
            try:
                root.attributes("-zoomed", True)  # MacOS
            except tk.TclError:
                # fallback: set manual ukuran sesuai layar
                w, h = root.winfo_screenwidth(), root.winfo_screenheight()
                root.geometry(f"{w}x{h}+0+0")

        # Kontrol atas dengan scroll horizontal
        control_canvas = tk.Canvas(root, height=100)
        control_scroll = tk.Scrollbar(
            root, orient="horizontal", command=control_canvas.xview
        )
        control_frame = tk.Frame(control_canvas)

        control_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(
                scrollregion=control_canvas.bbox("all")),
        )
        control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        control_canvas.configure(xscrollcommand=control_scroll.set)

        control_canvas.pack(fill="x", side="top")
        control_scroll.pack(fill="x", side="top")

        # === Kontrol satu baris ===
        tk.Label(control_frame, text="Clip Limit:").pack(side=tk.LEFT, padx=5)
        self.clip_var = tk.DoubleVar(value=2.0)
        self.clip_slider = tk.Scale(
            control_frame,
            from_=0.1,
            to=20.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.clip_var,
            command=self.on_slider_change,
        )
        self.clip_slider.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Tile Grid Size:").pack(
            side=tk.LEFT, padx=5)
        self.tile_var = tk.IntVar(value=8)
        self.tile_slider = tk.Scale(
            control_frame,
            from_=2,
            to=16,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=300,
            variable=self.tile_var,
            command=self.on_slider_change,
        )
        self.tile_slider.pack(side=tk.LEFT, padx=5)

        self.process_btn = tk.Button(
            control_frame,
            text="✅ Proses Semua Dataset",
            command=self.process_all,
            bg="green",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        self.process_btn.pack(side=tk.LEFT, padx=20)

        # Canvas grid dinamis
        canvas_frame = tk.Frame(root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.cells = []
        for i in range(GRID_ROWS):
            row = []
            for j in range(GRID_COLS):
                cell = tk.Frame(canvas_frame, relief=tk.RAISED, borderwidth=1)
                cell.grid(row=i, column=j, padx=5, pady=5, sticky="nsew")
                row.append(cell)
            self.cells.append(row)

        for i in range(GRID_ROWS):
            canvas_frame.grid_rowconfigure(i, weight=1)
        for j in range(GRID_COLS):
            canvas_frame.grid_columnconfigure(j, weight=1)

        self.update_previews()

    def on_slider_change(self, event=None):
        self.clip_limit = self.clip_var.get()
        self.tile_grid_size = self.tile_var.get()
        self.update_previews()

    def update_previews(self):
        self.photo_images.clear()
        for idx, img_path in enumerate(self.image_paths):
            i, j = divmod(idx, GRID_COLS)
            cell = self.cells[i][j]
            for widget in cell.winfo_children():
                widget.destroy()

            try:
                orig_pil = Image.open(img_path).convert("RGB")
                orig_pil.thumbnail(MAX_PREVIEW_SIZE, Image.LANCZOS)
                orig_tk = ImageTk.PhotoImage(orig_pil)

                clahe_pil = apply_clahe_to_image(
                    img_path, self.clip_limit, self.tile_grid_size
                )
                clahe_pil.thumbnail(MAX_PREVIEW_SIZE, Image.LANCZOS)
                clahe_tk = ImageTk.PhotoImage(clahe_pil)

                img_frame = tk.Frame(cell)
                img_frame.pack(expand=True)
                left_label = tk.Label(img_frame, image=orig_tk)
                left_label.grid(row=0, column=0, padx=5, pady=5)
                right_label = tk.Label(img_frame, image=clahe_tk)
                right_label.grid(row=0, column=1, padx=5, pady=5)

                self.photo_images.extend([orig_tk, clahe_tk])

                filename = os.path.basename(img_path)
                tk.Label(cell, text=filename[:20],
                         font=("Arial", 9)).pack(pady=3)

            except Exception as e:
                tk.Label(cell, text=f"Error: {e}", fg="red").pack()

    def process_all(self):
        clip = self.clip_limit
        tile = self.tile_grid_size
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        new_dataset_root = f"{DATASET_ROOT}_CLAHE_clip{clip}_tile{tile}_{timestamp}"

        print(f"📁 Membuat dataset: {new_dataset_root}")

        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(new_dataset_root,
                        "images", split), exist_ok=True)
            os.makedirs(os.path.join(new_dataset_root,
                        "labels", split), exist_ok=True)

        tasks = []
        total_images = 0
        for split in ["train", "val", "test"]:
            img_dir = os.path.join(DATASET_ROOT, "images", split)
            label_dir = os.path.join(DATASET_ROOT, "labels", split)
            new_img_dir = os.path.join(new_dataset_root, "images", split)
            new_label_dir = os.path.join(new_dataset_root, "labels", split)

            for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
                shutil.copy2(label_file, new_label_dir)

            for img_path in glob.glob(os.path.join(img_dir, "*.*")):
                filename = os.path.basename(img_path)
                out_path = os.path.join(new_img_dir, filename)
                tasks.append((img_path, out_path, clip, tile))
                total_images += 1

        print(
            f"🔄 Memproses {total_images} gambar dengan CLAHE (clip={clip}, tile={tile})...")
        with ProcessPoolExecutor() as executor:
            list(executor.map(save_clahe_image, tasks))

        print(f"✅ Selesai! Dataset disimpan di:\n{new_dataset_root}")
        messagebox.showinfo(
            "Selesai", f"Dataset CLAHE berhasil dibuat!\n\n{new_dataset_root}")


# === Jalankan GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = CLAHEPreviewApp(root, preview_images)
    root.mainloop()
