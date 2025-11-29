import os
import glob
import json
from PIL import Image
import yaml

# === CONFIG ===
DATASET_ROOT = "/home/dorisjuarsa/Apps/DorisjuarsaProgramTesis/Datasets/DorisjuarsaDatasetBalance360x360"
OUTPUT_JSON = os.path.join(DATASET_ROOT, "dataset_metadata.json")

# === UTILS (DIPINDAHKAN KE ATAS) ===


def get_file_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
        return "image"
    elif ext == ".txt":
        return "label"
    else:
        return "other"


# Load class names dari data.yaml
with open(os.path.join(DATASET_ROOT, "data.yaml"), "r") as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml["names"]

# === BUILD GLOBAL MAPPING: image_base_name -> image_index, label_base_name -> label_index ===
image_index_map = {}  # base_name -> index
label_index_map = {}  # base_name -> index

# Collect all images
image_files = []
for root, dirs, files in os.walk(os.path.join(DATASET_ROOT, "images")):
    for f in files:
        if get_file_type(f) == "image":
            full_path = os.path.join(root, f)
            base_name = os.path.splitext(f)[0]
            image_files.append((full_path, base_name))

# Assign image_index
for idx, (full_path, base_name) in enumerate(sorted(image_files, key=lambda x: x[0])):
    image_index_map[base_name] = idx

# Collect all labels
label_files = []
for root, dirs, files in os.walk(os.path.join(DATASET_ROOT, "labels")):
    for f in files:
        if get_file_type(f) == "label":
            full_path = os.path.join(root, f)
            base_name = os.path.splitext(f)[0]
            label_files.append((full_path, base_name))

# Assign label_index
for idx, (full_path, base_name) in enumerate(sorted(label_files, key=lambda x: x[0])):
    label_index_map[base_name] = idx

# === BUILD TREE FUNCTION ===


def build_tree(path, parent_name="", is_root=False):
    """Build tree recursively from path"""
    children = []

    # List all items in directory
    items = sorted(os.listdir(path))

    for item in items:
        full_path = os.path.join(path, item)

        if os.path.isfile(full_path):
            node = {
                "name": item,
                "type": "file",
                # relative path from root
                "path": os.path.relpath(full_path, DATASET_ROOT)
            }

            file_type = get_file_type(full_path)
            base_name = os.path.splitext(item)[0]

            if file_type == "image":
                try:
                    with Image.open(full_path) as img:
                        node["size"] = list(img.size)
                except Exception as e:
                    node["size"] = [0, 0]
                    node["error"] = str(e)

                # Assign image_index
                node["image_index"] = image_index_map.get(base_name, -1)

                # Cari label yang sesuai
                rel_dir = os.path.relpath(
                    os.path.dirname(full_path), DATASET_ROOT)
                if rel_dir.startswith("images/"):
                    label_dir = rel_dir.replace("images/", "labels/", 1)
                    label_candidate = os.path.join(
                        DATASET_ROOT, label_dir, base_name + ".txt")
                    if os.path.exists(label_candidate):
                        node["label_file"] = os.path.relpath(
                            label_candidate, DATASET_ROOT)
                        node["label_index"] = label_index_map.get(
                            base_name, -1)

            elif file_type == "label":
                objects = []
                try:
                    with open(full_path, "r") as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            cls_id, x, y, w, h = parts
                            cls_id = int(cls_id)
                            x, y, w, h = map(float, [x, y, w, h])
                            objects.append({
                                "class_id": cls_id,
                                "class_name": class_names[cls_id],
                                "bbox": [x, y, w, h]
                            })
                    node["objects"] = objects
                except Exception as e:
                    node["error"] = str(e)

                # Assign label_index
                node["label_index"] = label_index_map.get(base_name, -1)

                # Assign image_index jika ada pasangan
                if base_name in image_index_map:
                    node["image_index"] = image_index_map[base_name]

            children.append(node)

        elif os.path.isdir(full_path):
            dir_node = {
                "name": item,
                "type": "folder",
                "children": build_tree(full_path, item, is_root=False)
            }
            children.append(dir_node)

    return children


# === BUILD METADATA TREE ===
metadata_tree = {
    "name": os.path.basename(DATASET_ROOT),
    "type": "folder",
    "children": []
}

# Tambahkan data.yaml sebagai node pertama
data_yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
if os.path.exists(data_yaml_path):
    metadata_tree["children"].append({
        "name": "data.yaml",
        "type": "file",
        "path": "data.yaml",
        "content": data_yaml  # opsional: bisa dihapus jika terlalu besar
    })

# Tambahkan folder images/ dan labels/ secara rekursif
for folder_name in ["images", "labels"]:
    folder_path = os.path.join(DATASET_ROOT, folder_name)
    if os.path.exists(folder_path):
        folder_node = {
            "name": folder_name,
            "type": "folder",
            "children": build_tree(folder_path, folder_name, is_root=False)
        }
        metadata_tree["children"].append(folder_node)

# === SIMPAN JSON ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(metadata_tree, f, indent=2)

print(f"✅ JSON metadata berhasil dibuat: {OUTPUT_JSON}")
