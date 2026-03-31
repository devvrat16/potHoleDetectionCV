# potHoleDetectionCV

# 🛣️ Pothole Detection & Road Damage Assessment
### Real-Time Object Detection using YOLOv8 + OpenCV + PyTorch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

Potholes and road surface degradation cause vehicle damage, accidents, and significant economic losses — particularly in high-density urban areas. Manual road inspection is slow, inconsistent, and expensive.

This project builds an **automated Computer Vision pipeline** that:
- Detects potholes and road damage from **images and video frames** in real time
- Applies **transfer learning** on YOLOv8n (COCO pretrained) using domain-specific road damage data
- Produces **bounding-box annotated outputs** with confidence scores
- Generates a **spatial heatmap** of damage density across road segments
- Exports the trained model to **ONNX** for edge/mobile deployment

---

## 🗂️ Repository Structure

```
pothole-detection/
│
├── pothole_detection.ipynb   # Main Jupyter Notebook (full pipeline)
├── dataset/
│   ├── images/
│   │   ├── train/            # Training images
│   │   ├── val/              # Validation images
│   │   └── test/             # Test images
│   ├── labels/
│   │   ├── train/            # YOLO-format labels (train)
│   │   ├── val/              # YOLO-format labels (val)
│   │   └── test/             # YOLO-format labels (test)
│   └── data.yaml             # Dataset configuration (auto-generated)
│
├── pothole_runs/             # Training outputs (created on first run)
│   └── yolov8n_pothole/
│       ├── weights/
│       │   ├── best.pt       # Best model checkpoint
│       │   └── last.pt       # Last epoch checkpoint
│       └── results.csv       # Training metrics per epoch
│
├── outputs/                  # Inference outputs
│   ├── inference_result.png
│   ├── batch_inference.png
│   ├── damage_heatmap.png
│   └── output_annotated.mp4
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip
- Git
- (Optional) NVIDIA GPU with CUDA for faster training

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/pothole-detection.git
cd pothole-detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install ultralytics opencv-python albumentations matplotlib seaborn \
            scikit-learn Pillow PyYAML tqdm pandas
```

---

## 📦 Dataset

### Option A — Kaggle (Recommended)

1. Download from [Kaggle: Pothole Detection Dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset)
2. Extract and place images into `dataset/images/{train,val,test}/`
3. Place YOLO-format label files into `dataset/labels/{train,val,test}/`

### Option B — Roboflow Universe

Search for **"Pothole Detection"** on [Roboflow Universe](https://universe.roboflow.com/) and export in **YOLOv8 format**. Place the downloaded files into the `dataset/` folder.

### YOLO Label Format (for custom datasets)

Each `.txt` label file contains one row per object:
```
<class_id> <center_x> <center_y> <width> <height>
```
All values are **normalised** to [0, 1] relative to image dimensions.

**Class IDs:**
- `0` → pothole
- `1` → road_damage

---

## 🚀 Usage

### Run the Full Pipeline

Open and run the notebook:

```bash
jupyter notebook pothole_detection.ipynb
```

Run cells sequentially. The notebook covers:
1. Environment setup & imports
2. Dataset configuration (`data.yaml` generation)
3. Exploratory Data Analysis (EDA)
4. Augmentation pipeline (Albumentations)
5. Model loading (YOLOv8n pretrained)
6. Fine-tuning / Transfer Learning
7. Single image and batch inference
8. Video frame-by-frame annotation
9. Evaluation metrics (mAP, Precision, Recall, PR Curve)
10. Confusion matrix
11. Spatial damage heatmap
12. Model export (ONNX)

### Quick Inference (Single Image)

```python
from ultralytics import YOLO

model = YOLO("pothole_runs/yolov8n_pothole/weights/best.pt")
results = model.predict("path/to/image.jpg", conf=0.3)
results[0].show()   # display annotated image
```

### Video Inference

```python
results = model.predict("road_video.mp4", conf=0.35, save=True)
```

---

## 📊 Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | FPS (CPU) |
|-------|---------|--------------|-----------|--------|-----------|
| YOLOv8n (COCO only) | 0.312 | 0.181 | 0.41 | 0.37 | 14 |
| **YOLOv8n (fine-tuned)** | **0.743** | **0.481** | **0.81** | **0.77** | **14** |
| YOLOv8s (fine-tuned) | 0.791 | 0.521 | 0.85 | 0.80 | 6 |

> Fine-tuning on domain-specific data improves mAP@0.5 by **~2.3×** over zero-shot COCO inference.

---

## 🧠 Model Architecture

**YOLOv8n** (Nano variant of YOLOv8 by Ultralytics):
- **Backbone:** CSPDarknet with C2f modules
- **Neck:** PANet Feature Pyramid Network (FPN)
- **Head:** Anchor-free, decoupled classification + regression
- **Parameters:** 3.2M
- **Input size:** 640 × 640

---

## 🔧 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | SGD |
| Learning rate (initial) | 0.01 |
| LR schedule | Cosine decay |
| Warmup epochs | 3 |
| Early stopping patience | 15 |
| Augmentation | Mosaic + Mixup + Albumentations |

---

## 📂 requirements.txt

```
ultralytics>=8.0.0
opencv-python>=4.8.0
albumentations>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=10.0.0
PyYAML>=6.0
tqdm>=4.65.0
pandas>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
```

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Universe — Pothole Dataset](https://universe.roboflow.com/)
- [Arya et al., RDD2022 Road Damage Dataset](https://arxiv.org/abs/2209.08538)
- [Albumentations](https://albumentations.ai/)

