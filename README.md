<div align="center">

# 👗 Few-Shot Fashion Category Recognition in Emerging Apparel Markets

### Leveraging Meta-Learning and Part-Aware Feature Extraction for Robust Fashion Classification with Limited Training Data

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

<br>

</div>

---

## 📋 Overview

This project implements a **Few-Shot Learning** system for fashion apparel category recognition using the [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset. The system leverages advanced deep learning techniques including **ConvNeXt feature extraction**, **Part-Aware Pooling**, and **Prototypical Meta-Learning** to achieve robust classification even with limited training examples per category.

> [!NOTE]
> The entire pipeline — from data preprocessing to evaluation — is contained in a single notebook: [`CompletePipeLine.ipynb`](CompletePipeLine.ipynb).

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🧠 **Advanced Feature Extraction** | ConvNeXt-Tiny backbone producing 768-dim features for superior performance |
| 🔍 **Part-Aware Pooling** | Spatial attention mechanism focusing on discriminative regions of fashion items |
| 🔄 **Meta-Learning Framework** | Prototypical networks with episodic training for few-shot generalization |
| 📦 **End-to-End Pipeline** | Complete system from data preprocessing to model evaluation |
| ⚡ **GPU Accelerated** | CUDA-enabled training with memory optimization for efficient processing |

---

## 🏗️ Architecture

```
┌──────────────┐    ┌───────────────────────┐    ┌──────────────────┐    ┌──────────────────────┐    ┌───────────────────────┐
│ Input Images │───▶│ ConvNeXt-Tiny Feature │───▶│ Part-Aware       │───▶│ Prototypical         │───▶│ Category              │
│ (224×224)    │    │ Extraction (768-dim)  │    │ Pooling (768-dim)│    │ Meta-Learning (512-d)│    │ Classification        │
└──────────────┘    └───────────────────────┘    └──────────────────┘    └──────────────────────┘    └───────────────────────┘
```

> For detailed architecture diagrams and component descriptions, see [`Architecture.md`](Architecture.md).

---

## 📊 Dataset

**[DeepFashion2](https://github.com/switchablenorms/DeepFashion2)** — A large-scale fashion benchmark dataset:

| Split | Samples |
|:---:|:---:|
| **Training** | 312,186 |
| **Validation** | 52,490 |
| **Test** | 62,629 |

<details>
<summary><b>📁 13 Fashion Categories</b></summary>

| ID | Category | ID | Category |
|:--:|---|:--:|---|
| 1 | Short Sleeve Top | 8 | Trousers |
| 2 | Long Sleeve Top | 9 | Skirt |
| 3 | Short Sleeve Outwear | 10 | Short Sleeve Dress |
| 4 | Long Sleeve Outwear | 11 | Long Sleeve Dress |
| 5 | Vest | 12 | Vest Dress |
| 6 | Sling | 13 | Sling Dress |
| 7 | Shorts | | |

</details>

<details>
<summary><b>🗂️ Dataset Structure</b></summary>

```
DeepFashion2/
  deepfashion2_original_images/
    train/
      image/        ← 191,961 images (.jpg)
      annos/        ← 191,961 annotations (.json)
    validation/
      image/        ← 32,153 images (.jpg)
      annos/        ← 32,153 annotations (.json)
    test/
      test/image/   ← 62,629 images (.jpg)
  img_info_dataframes/
    train.csv
    validation.csv
    test.csv
```

</details>

Each sample includes high-resolution images, bounding boxes, segmentation masks (polygons), landmark keypoints, and category/style labels.

---

## 📈 Results

### 🏆 Overall Performance

> [!IMPORTANT]
> These are the **actual results** obtained from running the complete pipeline in `CompletePipeLine.ipynb`.

| Metric | Score |
|:---|:---:|
| **Best Meta-Learning Accuracy** | **96.67%** |
| **Final Validation Accuracy** | **76.74%** |
| **Precision (Weighted)** | **0.7618** |
| **Recall (Weighted)** | **0.7674** |
| **F1-Score (Weighted)** | **0.7604** |

### 📋 Per-Class Performance

| Category | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Short Sleeve Top | 0.828 | 0.885 | 0.855 | 12,556 |
| Long Sleeve Top | 0.715 | 0.630 | 0.669 | 5,966 |
| Short Sleeve Outwear | 0.667 | 0.249 | 0.289 | 142 |
| Long Sleeve Outwear | 0.761 | 0.670 | 0.712 | 2,011 |
| Vest | 0.694 | 0.489 | 0.564 | 2,113 |
| Sling | 0.650 | 0.355 | 0.431 | 322 |
| Shorts | 0.673 | 0.793 | 0.726 | 4,167 |
| **Trousers** | **0.896** | **0.936** | **0.916** | **9,586** |
| Skirt | 0.733 | 0.725 | 0.729 | 6,522 |
| Short Sleeve Dress | 0.646 | 0.685 | 0.665 | 3,127 |
| Long Sleeve Dress | 0.542 | 0.455 | 0.492 | 1,477 |
| Vest Dress | 0.702 | 0.760 | 0.729 | 3,352 |
| Sling Dress | 0.587 | 0.447 | 0.502 | 1,149 |

> [!TIP]
> **Trousers** achieved the highest F1-score (0.916), while **Short Sleeve Outwear** had the lowest (0.289) — largely due to having only 142 validation samples.

### ⚙️ Training Configuration

<table>
<tr>
<th>Meta-Learning Phase</th>
<th>Fine-Tuning Phase</th>
</tr>
<tr>
<td>

| Parameter | Value |
|:---|:---:|
| Episodes | 2,000 |
| N-way | 2 |
| K-shot | 13 |
| Query samples | 15 |
| Learning rate | 0.001 |
| Scheduler | Cosine Annealing |

</td>
<td>

| Parameter | Value |
|:---|:---:|
| Epochs | 25 |
| Batch size | 256 |
| Embedding LR | 0.0001 |
| Classifier LR | 0.001 |
| Optimizer | Adam |

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

<details>
<summary><b>📦 Required Packages</b></summary>

- PyTorch >= 2.0
- torchvision
- numpy
- pandas
- opencv-python
- scikit-learn
- matplotlib
- seaborn
- tqdm

</details>

### Running the Pipeline

The complete pipeline is implemented in [`CompletePipeLine.ipynb`](CompletePipeLine.ipynb). Execute the notebook cells sequentially:

```
1️⃣  Data Preprocessing     → Image resizing & annotation normalization
2️⃣  Feature Extraction      → ConvNeXt-based feature extraction
3️⃣  Part-Aware Pooling      → Spatial attention & feature fusion
4️⃣  Meta-Learning           → Episodic training & fine-tuning
5️⃣  Evaluation              → Comprehensive performance metrics & visualization
```

---

## 🔑 Key Components

### 1. Feature Extraction
- **Backbone**: ConvNeXt-Tiny (pretrained on ImageNet)
- **Output**: 768-dimensional feature vectors
- **Advantage**: Significant improvement over EfficientNet-B0 (54% → 85%+ accuracy)

### 2. Part-Aware Pooling
- Divides features into **5 horizontal regions**
- Applies weighted spatial pooling per part
- Fuses part features with attention mechanism
- Output: 768-dim fused features

### 3. Meta-Learning
- **Embedding Network**: 768 → 512 dimensions with skip connections
- **Prototypical Classifier**: Learnable class prototypes with temperature scaling
- **Training Strategy**: Episodic meta-learning followed by full dataset fine-tuning

---

## 📁 Project Structure

```
Few-shot-Fashion-Category-Recognition/
├── 📓 CompletePipeLine.ipynb        # Main notebook with complete pipeline
├── 📋 requirements.txt              # Python dependencies
├── 📖 README.md                     # This file
├── 📐 Architecture.md               # Detailed architecture documentation
├── 📁 DeepFashion2/                 # Dataset directory
│   ├── deepfashion2_original_images/
│   └── img_info_dataframes/
└── 📁 output/                       # Generated outputs
    ├── input/                       # Preprocessed data
    ├── resized/                     # Resized images
    ├── FEATURES_DIR/                # Extracted features
    ├── PartAwarePooling/            # PAP features
    ├── models/                      # Trained models
    │   ├── optimal_embedding_net.pt
    │   └── optimal_fewshot_model.pt
    └── results/                     # Evaluation results
        ├── optimal_fewshot_results.json
        ├── optimal_fewshot_training_curves.png
        └── optimal_fewshot_confusion_matrix.png
```

---

## 🔬 Technical Details

<details>
<summary><b>📷 Data Preprocessing</b></summary>

- **Resizing**: All images normalized to 224×224 pixels
- **Normalization**: ImageNet mean/std (transfer learning)
- **Augmentation**: Random crop, flip, rotation, color jitter (training only)
- **Annotations**: Bounding boxes and segmentation masks rescaled accordingly

</details>

<details>
<summary><b>💾 Memory Optimization</b></summary>

- Batch-wise feature processing to prevent OOM errors
- Memory-mapped arrays for large feature matrices
- Aggressive garbage collection and cache clearing
- Efficient numpy/torch tensor handling

</details>

<details>
<summary><b>🖥️ GPU Utilization</b></summary>

- CUDA-enabled training when available
- Automatic device detection and allocation
- Pinned memory for faster GPU transfers
- Mixed precision training support

</details>

---

## 📊 Model Outputs

The trained system produces:

| Output | Description |
|---|---|
| **Embedding Network** | Learned 512-dim embedding space |
| **Class Prototypes** | 13 category prototypes in embedding space |
| **Predictions** | Category labels with confidence scores |
| **Per-class Metrics** | Accuracy, Precision, Recall, F1-score per category |
| **Visualizations** | Training curves & confusion matrix |

---

## 🎓 Applications

- 🛒 **Fashion E-commerce** — Automated product categorization
- 👔 **Style Recommendation** — Similar item retrieval
- 🔎 **Visual Search** — Find products from images
- ✅ **Quality Control** — Detect mislabeled products
- 📊 **Trend Analysis** — Track fashion category popularity

---

## 📚 References

1. **DeepFashion2** — *A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images*
2. **Prototypical Networks** — *Prototypical Networks for Few-shot Learning* (Snell et al., 2017)
3. **ConvNeXt** — *A ConvNet for the 2020s* (Liu et al., 2022)
4. **Part-Based Models** — *Part-based R-CNNs for Fine-grained Category Detection*

---

### 🙏 Acknowledgments

*DeepFashion2 Dataset Creators • PyTorch & Torchvision Teams • Open-Source Community*

---

> **Note**: This project is part of academic research on few-shot learning applications in the fashion domain.

</div>
