# 🚁 Aerial Object Classifier

> Classify drones, birds, and airplanes from aerial images using deep learning.  
> **EfficientNetV2-S** fine-tuned on the [BirdVsDroneVsAirplane dataset](https://www.kaggle.com/datasets/maryamlsgumel/drone-detection-dataset) — achieving **97.49% test accuracy**.

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **97.49%** |
| Best Val Accuracy | **96.93%** |
| Model | EfficientNetV2-S |
| Epochs | 50 |
| Input Size | 224×224 |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Aeroplanes | 0.9718 | 0.9079 | 0.9388 |
| Birds | 0.9205 | 0.9310 | 0.9257 |
| Drones | 0.8915 | 0.9583 | 0.9237 |

---

## 🗂️ Dataset

- **Source:** [Kaggle — Drone Detection Dataset](https://www.kaggle.com/datasets/maryamlsgumel/drone-detection-dataset)
- **Classes:** Aeroplanes 🛩️ · Birds 🐦 · Drones 🚁
- **Split:** 70% train · 15% val · 15% test (stratified)

---

## 🧠 Model & Training

- **Architecture:** EfficientNetV2-S (pretrained on ImageNet via `timm`)
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingWarmRestarts (T0=15)
- **Augmentation:** RandomResizedCrop, Flips, Rotation, ColorJitter, GaussNoise, MotionBlur, CoarseDropout
- **Inference:** Test Time Augmentation (TTA, n=5)

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/goktani/aerial-object-classifier.git
cd aerial-object-classifier
pip install -r requirements.txt
```

### 2. Prepare Dataset
Download the dataset from Kaggle and place it as:
```
data/
└── BirdVsDroneVsAirplane/
    ├── Aeroplanes/
    ├── Birds/
    └── Drones/
```

### 3. Open the Notebook
```bash
jupyter notebook aerial_classification.ipynb
```

Update the `DATA_DIR` variable in **Cell 3** to point to your local dataset path:
```python
DATA_DIR = Path("data/BirdVsDroneVsAirplane")
```

Then run all cells top to bottom. The notebook covers the full pipeline:
training, evaluation, TTA inference, and confusion matrix visualization.

---

## 📁 Project Structure

```
aerial-object-classifier/
├── aerial_classification.ipynb   # Full pipeline — train, eval, TTA, visualize
├── requirements.txt
└── README.md
```

---

## 📊 Kaggle Notebook

Full experiment with visualizations available on Kaggle:  
👉 [Aerial Object Classification | EfficientNetV2-S | 97.49% Acc](https://www.kaggle.com/code/goktani/aerial-object-classification-efficientnetv2-s)

---

## 📜 License

MIT License — free to use, modify, and distribute.
