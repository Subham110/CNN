# Convolutional Neural Networks & Computer Vision 🧠

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.15+-red.svg)](https://keras.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Package Manager](https://img.shields.io/badge/Managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)

A collection of Deep Learning and Computer Vision projects covering **Convolutional Neural Networks (CNNs)**, **Transfer Learning**, and **Instance Segmentation (YOLOv8)** using TensorFlow, Keras, and Ultralytics.

---

## 📌 Features

### 1. 🕳️ Pothole Instance Segmentation (`pothole.py`)
- **Model**: Pre-trained `yolov8n-seg.pt` (YOLOv8 Nano Segmentation).
- **Dataset Analysis & EDA**:
  - Parses YOLO format `data.yaml` configuration.
  - Verifies image integrity and dimension uniformity across training and validation sets.
  - Visualizes random samples from the dataset with Matplotlib.
- **Training & Fine-Tuning**:
  - Trains for 80 epochs with early stopping (`patience=10`), `imgsz=640`, and `batch=16`.
  - Employs `AdamW` optimizer (`lr0=0.0001`, `lrf=0.01`), with `dropout=0.25` for regularization.
- **Evaluation & Diagnostics**:
  - Extracts and plots multi-component learning curves from `results.csv`:
    - **Bounding Box Loss** (`train/box_loss` vs `val/box_loss`)
    - **Classification Loss** (`train/cls_loss` vs `val/cls_loss`)
    - **Distribution Focal Loss** (`train/dfl_loss` vs `val/dfl_loss`)
    - **Segmentation Mask Loss** (`train/seg_loss` vs `val/seg_loss`)
- **Inference & Video Pipeline**:
  - Batch validation inference displaying predicted masks and bounding boxes in a 3x3 grid.
  - Video inference on road footage with OpenCV post-processing (`cv2.VideoWriter` / `mp4v` codec) for seamless video playback.

---

### 2. 🐱🐶 Cat vs Dog Classification (`cat_dog.py`)
- Deep CNN architecture with:
  - 3 Convolutional blocks (`Conv2D` + `BatchNormalization` + `MaxPooling2D`).
  - Fully connected classification head with `Dropout(0.1)` regularization.
- Data loading with `tf.keras.utils.image_dataset_from_directory` on 256x256 RGB images.
- Training accuracy/loss visualization and single-image test prediction.

---

### 3. 🔄 Transfer Learning with VGG16 (`Transfer.py`)
- Uses pre-trained **VGG16** (ImageNet weights) as a frozen convolutional feature extractor (`conv_base.trainable = False`).
- Custom classification head with `Dense(256, relu)` and `Dense(1, sigmoid)`.
- Real-time data augmentation via `ImageDataGenerator` (shear, zoom, horizontal flip).

---

### 4. ✍️ MNIST Digit Classification (`Train_CNN.py`)
- Baseline CNN for handwritten digit recognition on 28x28 grayscale images.
- Evaluates test accuracy (>98%) and plots training vs. validation accuracy.
- Performs single-sample inference and visualization.

---

### 5. 📓 Interactive Notebooks (`colab.ipynb`)
- Google Colab notebook for experimenting with YOLO object detection, model checkpoints, and datasets.

---

## 🏗️ Architectures

### MNIST CNN (`Train_CNN.py`)
```text
Input (28x28x1 Grayscale Image)
       │
       ▼
Conv2D (32 filters, 3x3 kernel, ReLU)
       │
       ▼
Conv2D (64 filters, 3x3 kernel, ReLU)
       │
       ▼
MaxPooling2D (2x2 pool size)
       │
       ▼
Flatten
       │
       ▼
Dense (128 units, ReLU activation)
       │
       ▼
Dense (10 units, Softmax activation) -> Output Probabilities (0-9)
```

### Pothole Segmentation Pipeline (`pothole.py`)
```text
Road Imagery / Video Stream
       │
       ▼
YOLOv8 Nano Segmentation Backbone & Neck (CSPDarknet + PANet)
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
Bounding Box Head         Classification Head       Mask Prototype & Coeff Head
(Box Coordinates)         (Pothole Class Conf)      (Polygon Segmentation Mask)
       │                         │                         │
       └─────────────────────────┼─────────────────────────┘
                                 │
                                 ▼
                     Annotated Image / Video
```

---

## 📁 Repository Structure

```text
CNN/
├── pothole.py            # YOLOv8 road pothole instance segmentation & video processing
├── Transfer.py           # VGG16 transfer learning with data augmentation
├── cat_dog.py            # Cat vs Dog classification with Batch Normalization & Dropout
├── Train_CNN.py          # MNIST CNN training, evaluation & inference script
├── main.py               # Application entrypoint
├── colab.ipynb           # Interactive Jupyter / Google Colab notebook
├── cat2.webp             # Sample cat image for inference testing
├── dog2.jpg              # Sample dog image for inference testing
├── pyproject.toml        # Project dependencies and configuration
├── uv.lock               # Deterministic dependency lockfile
├── .python-version       # Target Python version (3.13)
├── .gitignore            # Git ignore rules (includes Kaggle credential protection)
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- NVIDIA GPU with CUDA drivers (recommended for YOLOv8 training)

### 1. Clone the Repository

```bash
git clone https://github.com/Subham110/CNN.git
cd CNN
```

### 2. Environment Setup & Dependencies

#### Using `uv` (Recommended):

```bash
uv sync
```

#### Or using standard `pip`:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r pyproject.toml
```

Key dependencies:
- `ultralytics`
- `tensorflow`
- `keras`
- `opencv-python`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `pillow`
- `numpy`
- `pandas`
- `kagglehub`

---

## 💻 Usage

### 1. Road Pothole Segmentation
Ensure your dataset path in `pothole.py` points to your extracted YOLO segmentation dataset (containing `data.yaml`, `train/images`, `valid/images`), then run:

```bash
python pothole.py
```

### 2. Cat vs Dog Classification
```bash
python cat_dog.py
```

### 3. VGG16 Transfer Learning
```bash
python Transfer.py
```

### 4. Train the MNIST CNN Model
```bash
python Train_CNN.py
```

---

## 📊 Results Summary

| Task | Model | Input Size | Key Metrics / Objectives |
| :--- | :--- | :--- | :--- |
| **Pothole Segmentation** | YOLOv8n-seg | 640x640 | Mask & Box mAP, real-time video detection |
| **Cat vs Dog** | Custom CNN (3 Conv blocks) | 256x256 | Binary Crossentropy, Regularization (BN + Dropout) |
| **Pet Classification** | VGG16 (Transfer Learning) | 150x150 | Feature extraction with pre-trained ImageNet weights |
| **Digit Recognition** | Baseline CNN | 28x28 | >98% Test Accuracy on MNIST |

---

## 📄 License

This repository is maintained for educational and machine learning research purposes.
