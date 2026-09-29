# Convolutional Neural Networks (CNN) 🧠

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.15+-red.svg)](https://keras.io/)
[![Package Manager](https://img.shields.io/badge/Managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)

A collection of Deep Learning experiments, implementations, and workflows focusing on **Convolutional Neural Networks (CNNs)** for computer vision tasks using TensorFlow and Keras.

---

## 📌 Features

- **MNIST Digit Classification (`Train_CNN.py`)**:
  - Preprocesses 28x28 grayscale handwritten digits.
  - Builds and trains a multi-layer CNN architecture.
  - Evaluates model performance and plots training vs. validation accuracy.
  - Performs sample inference and visualizes predictions.
- **Cat vs Dog Image Classification (`cat_dog.py`)**:
  - Image classification pipeline for binary pet recognition using Kaggle datasets.
- **Object Detection & Exploration (`colab.ipynb`)**:
  - Google Colab-ready notebook for advanced computer vision experiments (including YOLO and custom detection).

---

## 🏗️ Model Architecture (MNIST)

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

---

## 📁 Repository Structure

```text
CNN/
├── Train_CNN.py          # MNIST CNN training, evaluation & inference script
├── cat_dog.py            # Cat vs Dog classification pipeline
├── main.py               # Application entrypoint
├── colab.ipynb           # Interactive Jupyter / Google Colab notebook
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
- `tensorflow`
- `keras`
- `matplotlib`
- `numpy`
- `pandas`
- `kagglehub`

---

## 💻 Usage

### Train the MNIST CNN Model

Run the training pipeline to train the model, inspect accuracy metrics, and plot the training history:

```bash
python Train_CNN.py
```

### Run Application Entrypoint

```bash
python main.py
```

### Kaggle Datasets Configuration (for `cat_dog.py`)

If downloading datasets from Kaggle via `kagglehub` or the Kaggle API:
1. Download your `kaggle.json` token from [Kaggle Account Settings](https://www.kaggle.com/settings).
2. Place it in `~/.kaggle/kaggle.json` (or your local directory).
> **Note**: `kaggle.json` is already excluded in `.gitignore` to prevent leaking private credentials.

---

## 📊 Results

The baseline MNIST CNN model achieves high accuracy (>98%) within just 2 epochs of training using the Adam optimizer and sparse categorical crossentropy loss.

---

## 📄 License

This repository is maintained for educational and machine learning research purposes.

