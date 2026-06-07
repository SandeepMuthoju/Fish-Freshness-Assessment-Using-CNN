# 🐟 Fish Freshness Assessment Using CNN

**Machine Learning Course – Phase 2 Project**

---

## Project Overview
This project implements a Convolutional Neural Network (CNN) pipeline to automatically classify fish images into freshness categories. The goal is to replace subjective human inspection with an objective, scalable deep-learning solution.

---

## Dataset
| Field | Details |
|-------|---------|
| Source | [Freshness Fish Dataset – Kaggle](https://www.kaggle.com/datasets/atcharachumpol/freshness-fish-dataset) |
| Task | Multi-class image classification |
| Input size | 224 × 224 × 3 (RGB) |
| Split | 80 % train / 20 % validation |

---

## Models Implemented
| Model | Strategy |
|-------|---------|
| **Custom CNN** | 5 Conv blocks, batch norm, dropout |
| **MobileNetV2** | ImageNet feature extraction |
| **MobileNetV2 Fine-Tuned** | Top-55 layers unfrozen, lr = 1e-5 |
| **VGG16** | ImageNet feature extraction |

---

## Project Structure
```
fish-freshness-cnn/
├── fish_freshness_cnn.py       ← Main notebook / script (all 18 cells)
├── Fish_Freshness_CNN_Proposal.docx  ← Full project proposal
├── README.md
└── outputs/                    ← Generated figures (after running)
    ├── class_distribution.png
    ├── sample_images.png
    ├── custom_cnn_curves.png
    ├── mobilenetv2_curves.png
    ├── *_confusion_matrix.png
    ├── *_roc.png
    ├── gradcam_*.png
    ├── error_analysis.png
    └── model_comparison.png
```

---

## How to Run on Kaggle (Step-by-Step)

### Step 1 – Create a Kaggle account
Go to [kaggle.com](https://kaggle.com) and sign up (free).

### Step 2 – Add the dataset
1. Open the dataset page:
   [Freshness Fish Dataset](https://www.kaggle.com/datasets/atcharachumpol/freshness-fish-dataset)
2. Click **"Copy & Edit"** or go to your own notebook and click **"Add Data"** → search **"freshness fish dataset"** → add it.

### Step 3 – Create a new notebook
1. Go to **kaggle.com → Code → New Notebook**.
2. Choose **Python** notebook type.
3. Enable **GPU**: Settings → Accelerator → **GPU T4 x2** (free).

### Step 4 – Copy the code
1. Open `fish_freshness_cnn.py` from this repository.
2. Copy the entire content into your Kaggle notebook (paste into a single code cell, or split by the `# ── CELL N ──` comments into separate cells — recommended for easier debugging).

### Step 5 – Verify the dataset path
At the top of **CELL 2**, confirm:
```python
DATASET_DIR = '/kaggle/input/freshness-fish-dataset/Freshness Fish Dataset'
```
To check the exact path, run this in a new cell first:
```python
import os
for root, dirs, files in os.walk('/kaggle/input'):
    print(root)
    break
```
Adjust `DATASET_DIR` if the path differs.

### Step 6 – Run all cells
Click **Run All** (▶▶) or press **Shift+Enter** on each cell in order.

### Step 7 – View outputs
All figures are saved to `/kaggle/working/outputs/`. They are also visible inline in the notebook. After training completes, download them via **Output** tab on the right.

### Step 8 – Save the notebook
Click **Save Version** → **Save & Run All (Commit)** to make it public and get a shareable Kaggle URL.

---

## Requirements
All packages are pre-installed on Kaggle. For local use:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

---

## Outputs Generated
- Class distribution chart
- Sample augmented images
- Training / validation accuracy & loss curves (all models)
- Confusion matrices
- ROC curves (one-vs-rest)
- Grad-CAM heatmap visualisations
- Error analysis (misclassified samples)
- Model comparison bar chart
- Classification reports (CSV)

---

## References
- [Freshness Fish Dataset](https://www.kaggle.com/datasets/atcharachumpol/freshness-fish-dataset)
- [Reference Notebook](https://www.kaggle.com/code/killa92/acc-1-0-fish-freshness-classification-project/notebook)
- Keras Applications: https://keras.io/api/applications/
- Selvaraju et al. (2017) – Grad-CAM: Visual Explanations from Deep Networks
