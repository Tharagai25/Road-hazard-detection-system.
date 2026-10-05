# 🛣️ RoadWatch AI — Road Damage & Pothole Detection System

A beginner-friendly, portfolio-quality Python application using **Streamlit**, **OpenCV**, and **Ultralytics YOLO** to detect, annotate, and analyze potholes and road surface cracks in images and dashcam video footage.

---

## 🌟 Overview & Key Features

- **📸 Image Detection Mode**: Upload JPEG/PNG images, detect surface defects, view annotated bounding boxes with confidence scores, inspect damage statistics, and export detection logs to CSV.
- **🎥 Video Detection Mode**: Process MP4/MOV/AVI video footage frame-by-frame with real-time progress indicators, FPS tracking, and video defect summaries.
- **⚡ Lightweight & Fast**: Designed to run seamlessly on CPU or CUDA-enabled GPUs using Ultralytics YOLO models (e.g., `yolo11n.pt`).
- **🛡️ Clean & Safe**: Processing is performed locally; uploaded media is never stored permanently.

---

## 📁 Repository Structure

```text
RoadWatch-AI/
├── app.py              # Streamlit interface and image/video inference pipeline
├── train.py            # Model fine-tuning script for YOLO models
├── requirements.txt    # Required Python dependencies
├── .gitignore          # Excludes datasets, model weights, virtual environments & temp files
└── README.md           # Setup guide, dataset setup, training & app usage instructions
```

> **Note**: Model weights (`models/best.pt`) and dataset files are not committed to Git. Follow the setup instructions below to train your model locally.

---

## 🏷️ Road Damage Classes & Datasets

### Primary Dataset: RDD2022 (Road Damage Dataset 2022)
RDD2022 is a multi-national dataset containing road images collected across Japan, India, the Czech Republic, Norway, the United States, and China.

- **Classes**:
  - `D00`: Longitudinal cracks
  - `D10`: Transverse cracks
  - `D20`: Alligator cracks
  - `D40`: Potholes
- **Dataset References**:
  - **Paper**: [RDD2022 Research Paper (arXiv:2209.08538)](https://arxiv.org/abs/2209.08538)
  - **Download Page**: [Figshare RDD2022 Dataset Release](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)
  - **Project Repository**: [Sekilab RoadDamageDetector GitHub](https://github.com/sekilab/RoadDamageDetector)

### Secondary Dataset: Roboflow Pothole Dataset (Quick Experiment)
For quick single-class testing:
- **Link**: [Roboflow Public Pothole Dataset](https://public.roboflow.com/object-detection/pothole)
- **Details**: 665 images with bounding box annotations.
- **License**: Open Data Commons Open Database License (ODbL v1.0).

> ⚠️ **Dataset Disclaimer**: Do not claim the model detects debris, fallen branches, or road hazards unless those specific categories are present in your training data.

---

## 📂 Dataset Preparation & Expected Folder Layout

Organize your downloaded dataset into standard YOLO object detection format:

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Sample `data.yaml` Configuration File

Create a `data.yaml` file inside your dataset folder:

```yaml
# data.yaml template for RDD2022
path: ./dataset  # dataset root dir
train: images/train
val: images/val

# number of classes
nc: 4

# class names
names:
  0: D00  # Longitudinal Crack
  1: D10  # Transverse Crack
  2: D20  # Alligator Crack
  3: D40  # Pothole
```

*(For single-class datasets like Roboflow Pothole, set `nc: 1` and `names: {0: pothole}`).*

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Clone the repository, create a Python virtual environment, and install dependencies:

```bash
# Clone repository
git clone https://github.com/your-username/RoadWatch-AI.git
cd RoadWatch-AI

# Create virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Create virtual environment (Linux / macOS)
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

### 2. Train the Model

Fine-tune a lightweight YOLO model (`yolo11n.pt`) on your prepared dataset:

```bash
# Default training run (25 epochs, 640 image resolution)
python train.py --data path/to/dataset/data.yaml --epochs 25 --imgsz 640 --batch 16
```

#### Training Script Options
- `--data`: Path to your dataset `data.yaml` file (required).
- `--model`: Base YOLO model checkpoint (default: `yolo11n.pt`).
- `--epochs`: Number of training epochs (default: `25`).
- `--imgsz`: Input image size in pixels (default: `640`).
- `--batch`: Batch size (default: `16`).
- `--output`: Target location to save best model (default: `models/best.pt`).

Upon completion, `train.py` copies the best trained checkpoint to `models/best.pt`.

---

### 3. Run the Streamlit Web App

Start the interactive dashboard:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

If `models/best.pt` is present, the app loads it automatically. You can also upload custom PyTorch `.pt` model files via the app's sidebar.

---

## ⚠️ Important Limitations & Disclaimers

1. **Environmental Variability**: Model detection accuracy varies depending on camera quality, mounting angle, weather conditions (rain, shadows, snow), wet road surfaces, and country-specific road materials.
2. **Prototype & Research Purpose**: **RoadWatch AI** is a prototype designed for reporting, educational demonstration, and research. It should not be used as an official safety inspection system or for legal/municipal infrastructure determinations.
3. **Ultralytics Licensing**: Ultralytics YOLO models are subject to AGPL-3.0 licensing terms. Check the [Ultralytics Licensing Guide](https://ultralytics.com/license) if you plan to use this software for commercial purposes.

---

## 📤 Publishing to GitHub

To push this repository to GitHub:

```bash
git init
git add .
git commit -m "Initial commit of RoadWatch AI"
git branch -M main
git remote add origin https://github.com/your-username/RoadWatch-AI.git
git push -u origin main
```

*(Ensure dataset folders and `.pt` model files remain ignored by `.gitignore` before committing).*

---

## 📜 License & Attributions

- **Code Architecture**: MIT License
- **Ultralytics YOLO**: AGPL-3.0 / Enterprise License
- **RDD2022 Dataset**: Sekilab CRDDC 2022 / Figshare
- **Roboflow Pothole Dataset**: ODbL v1.0 License
