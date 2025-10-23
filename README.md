# 🎨 Few-Shot Fashion Category Recognition in Emerging Apparel Markets

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

**A complete deep learning pipeline for fashion image classification using meta-learning and part-aware spatial features**

[Overview](#-overview) • [Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Architecture](#-architecture)

</div>

---

## 📋 Overview

This project implements an end-to-end pipeline for fashion image classification using the **DeepFashion2** dataset. The system combines transfer learning (ResNet-50), part-aware spatial pooling, and few-shot meta-learning to achieve robust classification performance with limited labeled data.

### 🎯 Key Objectives

- ✅ **Few-shot learning** for emerging fashion categories with limited samples
- ✅ **Part-aware features** for fine-grained garment understanding
- ✅ **Transfer learning** leveraging pre-trained ResNet-50
- ✅ **Scalable pipeline** processing 427K+ fashion images

### 📊 Results Snapshot

| Metric | Value |
|--------|-------|
| **Meta-Learning Accuracy** | 72.0% (5-way 10-shot) |
| **Validation F1-Score** | 52.0% (13-class) |
| **Dataset Size** | 427,305 images |
| **Categories** | 13 fashion classes |

---

## 🔑 Key Features

- 🖼️ **Automated preprocessing** with annotation adjustment and augmentation
- 🧠 **ResNet-50 feature extraction** with ImageNet pre-training
- 🎯 **Part-aware spatial pooling** for garment-specific features
- 🚀 **Two-phase meta-learning**: Episodic training + fine-tuning
- 📈 **Prototypical networks** with learnable class prototypes
- ⚡ **Memory-efficient** batch processing for large datasets
- 📊 **Comprehensive evaluation** with per-class metrics and visualizations

---

## 📓 Project Structure

The pipeline consists of 4 Jupyter notebooks executed sequentially:

```
📦 Project
├── 📓 preprocessing.ipynb              # Stage 1: Image & annotation preprocessing
├── 📓 Feature_extraction.ipynb         # Stage 2: ResNet-50 feature extraction
├── 📓 PartAwarePooling.ipynb          # Stage 3: Spatial part-aware pooling
└── 📓 OptimalFewShotLearning.ipynb    # Stage 4: Meta-learning & classification
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (minimum 4GB VRAM, tested on GTX 1650)
- 16GB+ RAM
- 50GB+ free disk space

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/VISHAL-Nk/Few-shot-Fashion-Category-Recognition-in-Emerging-Apparel-Markets.git
cd Few-shot-Fashion-Category-Recognition-in-Emerging-Apparel-Markets
```

2. **Create virtual environment**
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn opencv-python
pip install scikit-learn tqdm jupyter ipykernel
pip install kagglehub  # For dataset download
```

4. **Download DeepFashion2 dataset**

Option A: Using Kaggle
```python
import kagglehub
dataset_path = kagglehub.dataset_download("thusharanair/deepfashion2-original-with-dataframes")
```

Option B: Manual download from [DeepFashion2 website](https://github.com/switchablenorms/DeepFashion2)

---

## 📖 Usage

### Step 1: Preprocessing (`preprocessing.ipynb`)

**Purpose**: Resize images to 224×224, adjust annotations, apply data augmentation

**Key Operations**:
- Image resizing with OpenCV
- Bounding box and segmentation rescaling
- Training augmentations: RandomCrop, Flip, Rotation, ColorJitter
- ImageNet normalization

**Outputs**:
- `output/resized/` - Resized images (train/val/test)
- `output/input/*.csv` - Updated annotation CSVs

**Runtime**: ~30 minutes (CPU-bound)

```python
# Run all cells in preprocessing.ipynb
# Configure paths to your dataset location
ROOT_DIR = "/path/to/your/output/"
dataset_path = "/path/to/DeepFashion2/"
```

---

### Step 2: Feature Extraction (`Feature_extraction.ipynb`)

**Purpose**: Extract 2048-dimensional features using pre-trained ResNet-50

**Key Operations**:
- Load pre-trained ResNet-50 (ImageNet weights)
- Remove final FC layer (use global avg pooling)
- Batch inference on preprocessed images
- Save feature vectors as `.npy` files

**Architecture**:
```
Input [224×224×3] → ResNet-50 → Global Avg Pool → Output [2048]
```

**Outputs**:
- `output/FEATURES_DIR/train_features.npy` - [312186, 2048]
- `output/FEATURES_DIR/val_features.npy` - [52490, 2048]
- `output/FEATURES_DIR/test_features.npy` - [62629, 2048]

**Runtime**: ~2 hours (GPU-accelerated)

```python
# Configuration
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

### Step 3: Part-Aware Pooling (`PartAwarePooling.ipynb`)

**Purpose**: Decompose spatial features into garment parts (collar, sleeves, body, etc.)

**Key Operations**:
1. Simulate spatial feature maps [N, 2048, 7, 7]
2. Generate part masks from segmentation polygons
3. Apply weighted spatial pooling per part
4. Fuse part features via average/attention

**Algorithm**:
```python
for each part k in [collar, upper_body, mid_body, lower_body, hemline]:
    f_k = Σ(mask_k * feature_map) / Σ(mask_k)
    
f_fused = mean(f_1, f_2, f_3, f_4, f_5)
```

**Outputs**:
- `output/PartAwarePooling/train_part_features.npy` - [312186, 5, 2048]
- `output/PartAwarePooling/train_fused_features.npy` - [312186, 2048]
- Part masks cached as `.pt` files for reuse

**Runtime**: ~45 minutes (mixed CPU/GPU)

---

### Step 4: Optimal Few-Shot Learning (`OptimalFewShotLearning.ipynb`)

**Purpose**: Train meta-learning model for few-shot classification

**Two-Phase Training**:

#### **Phase 1: Meta-Learning (Episodic Training)**
- Sample N-way K-shot episodes (5-way, 10-shot)
- Compute prototypes from support set
- Classify query set using prototypes
- Update embedding network

```python
# Hyperparameters
N_WAY = 5          # Number of classes per episode
K_SHOT = 10        # Support examples per class
Q_QUERY = 15       # Query examples per class
EPISODES = 1500    # Total training episodes
```

**Result**: 72% episode accuracy

#### **Phase 2: Fine-Tuning (Full Dataset)**
- Initialize prototypes from training data
- Train embedding network + prototypical classifier jointly
- Use differential learning rates (embedding: 0.0001, classifier: 0.001)

```python
# Architecture
Input [2048] → Embedding [512] → Prototypical Classifier → Output [13]
```

**Result**: 52% validation F1-score

**Outputs**:
- `output/models/optimal_embedding_net.pt` - Best embedding network
- `output/models/optimal_fewshot_model.pt` - Complete trained model
- `output/results/optimal_fewshot_results.json` - Performance metrics
- Training curves and confusion matrix visualizations

**Runtime**: ~3.5 hours (GPU-accelerated)

---

## 📊 Results

### Overall Performance

| Metric | Value | Context |
|--------|-------|---------|
| **Meta-Learning Accuracy** | **72.0%** | 5-way 10-shot episodes |
| **Validation Accuracy** | **53.0%** | 13-class full dataset |
| **Validation F1-Score** | **0.520** | Weighted average |
| **Precision** | 0.516 | Weighted average |
| **Recall** | 0.530 | Weighted average |

### Per-Class F1-Scores

| Category | F1-Score | Performance |
|----------|----------|-------------|
| 🏆 Trousers | 0.704 | Excellent |
| ✅ Short sleeve top | 0.613 | Good |
| ✅ Skirt | 0.488 | Moderate |
| ✅ Shorts | 0.477 | Moderate |
| ⚠️ Sling | 0.161 | Challenging |
| ⚠️ Short sleeve outwear | 0.013 | Very difficult (class imbalance) |

### Training Progression

**Phase 1 (Meta-Learning)**:
- Initial: ~20% (random 5-way baseline)
- Final: 72%
- Convergence: ~1000 episodes

**Phase 2 (Fine-Tuning)**:
- Initial F1: 0.35
- Final F1: 0.52
- Best epoch: 18/25

### Key Insights

✅ **Meta-learning advantage**: +17.6% accuracy over direct classification  
✅ **Few-shot capability**: Strong generalization to unseen episode configurations  
⚠️ **Class imbalance**: Performance correlates with training sample count  
⚠️ **Fine-grained confusion**: Similar categories (vest vs. vest dress) challenging  

---

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Preprocessing (preprocessing.ipynb)                   │
│  • Resize images: Variable → 224×224                            │
│  • Adjust annotations: bbox, segmentation, landmarks            │
│  • Apply augmentation: Crop, Flip, Rotate, ColorJitter         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Feature Extraction (Feature_extraction.ipynb)         │
│  • ResNet-50 (ImageNet pre-trained)                             │
│  • Global Average Pooling                                       │
│  • Output: 2048-D feature vectors                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Part-Aware Pooling (PartAwarePooling.ipynb)          │
│  • Simulate spatial features: [N, 2048, 7, 7]                  │
│  • Generate 5-part masks: collar/upper/mid/lower/hemline       │
│  • Weighted pooling per part                                    │
│  • Fusion: Average/Attention → [N, 2048]                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Few-Shot Learning (OptimalFewShotLearning.ipynb)     │
│  Phase 1: Meta-Learning (1500 episodes)                        │
│    • Sample N-way K-shot tasks                                  │
│    • Learn embedding network                                    │
│    • Result: 72% episode accuracy                              │
│                                                                  │
│  Phase 2: Fine-Tuning (25 epochs)                              │
│    • Initialize prototypical classifier                         │
│    • Joint training on full dataset                            │
│    • Result: 52% validation F1                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Model Components

#### Embedding Network
```
Input [2048] 
    ↓
Linear(2048 → 1024) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear(1024 → 768) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear(768 → 512) + BatchNorm + ReLU
    ↓
L2 Normalize → Output [512]
```

#### Prototypical Classifier
```
Embeddings [B, 512] → L2 Normalize
                        ↓
    Cosine Similarity with Prototypes [13, 512]
                        ↓
            Temperature Scaling (learnable)
                        ↓
                Logits [B, 13]
```

---

## 🎓 Dataset

**DeepFashion2** - Consumer-to-Shop Clothes Retrieval

### Statistics

| Split | Samples | Annotations | Labels |
|-------|---------|-------------|--------|
| Train | 312,186 | ✓ (segmentation, bbox, landmarks) | ✓ (13 categories) |
| Validation | 52,490 | ✓ | ✓ |
| Test | 62,629 | ✗ | ✗ |
| **Total** | **427,305** | - | - |

### Categories (13 classes)

1. Short sleeve top
2. Long sleeve top
3. Short sleeve outwear
4. Long sleeve outwear
5. Vest
6. Sling
7. Shorts
8. Trousers
9. Skirt
10. Short sleeve dress
11. Long sleeve dress
12. Vest dress
13. Sling dress

---

## 💡 Key Techniques

### 1. Transfer Learning
- **Pre-trained ResNet-50** on ImageNet (1.2M images)
- **Frozen backbone** for feature extraction (no fine-tuning)
- **Domain adaptation** from general objects → fashion items

### 2. Part-Aware Spatial Pooling
- **Motivation**: Different garment parts (collar, sleeves, hemline) have distinct features
- **Approach**: Divide spatial features into 5 horizontal parts
- **Benefit**: Preserves fine-grained spatial information vs. global pooling

### 3. Meta-Learning (Learning to Learn)
- **Algorithm**: Prototypical Networks
- **Training**: Episodic sampling of N-way K-shot tasks
- **Advantage**: Learns generalizable representations for few-shot scenarios
- **Result**: 72% accuracy on unseen 5-way 10-shot episodes

### 4. Two-Phase Training
- **Phase 1**: Meta-learning with small episodes → good embeddings
- **Phase 2**: Fine-tuning on full dataset → leverage all data
- **Synergy**: Combines few-shot generalization + supervised accuracy

### 5. Prototypical Classification
- **Concept**: Each class has a learnable prototype vector
- **Similarity**: Cosine distance in normalized embedding space
- **Calibration**: Temperature scaling for confidence adjustment

---

## ⚙️ Hardware Requirements

### Minimum (Tested Configuration)
- **GPU**: NVIDIA GTX 1650 (4GB VRAM)
- **RAM**: 16GB
- **Storage**: 50GB free space
- **CPU**: 4+ cores

### Recommended
- **GPU**: NVIDIA RTX 3060+ (12GB VRAM) → Larger batch sizes
- **RAM**: 32GB → Load full dataset in memory
- **Storage**: 100GB SSD → Faster I/O
- **CPU**: 8+ cores → Parallel preprocessing

### Execution Time (GTX 1650 4GB)
| Stage | Notebook | Time |
|-------|----------|------|
| 1 | `preprocessing.ipynb` | ~30 min |
| 2 | `Feature_extraction.ipynb` | ~2 hours |
| 3 | `PartAwarePooling.ipynb` | ~45 min |
| 4 | `OptimalFewShotLearning.ipynb` | ~3.5 hours |
| **Total** | **End-to-end** | **~6.5 hours** |

---

## 🔬 Future Work

### Immediate Improvements
1. **Integrate part-aware features** into few-shot learning (currently uses global features)
2. **Address class imbalance** with focal loss, SMOTE, or balanced sampling
3. **Attention-based part fusion** instead of average pooling
4. **Hyperparameter optimization** (grid search, Bayesian optimization)

### Research Directions
5. **Cross-domain transfer**: DeepFashion2 → emerging market datasets
6. **Multi-modal learning**: Image + text descriptions
7. **Contrastive pre-training**: Triplet loss, SimCLR
8. **Hierarchical classification**: Coarse-to-fine category prediction

### Production Deployment
9. **Model compression**: Quantization (INT8), pruning, knowledge distillation
10. **Real-time inference**: ONNX export, TensorRT optimization
11. **Active learning**: Deploy → collect uncertain predictions → retrain

---

## 📚 References

### Papers
- **DeepFashion2**: Ge et al. "DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images." CVPR 2019
- **ResNet**: He et al. "Deep Residual Learning for Image Recognition." CVPR 2016
- **Prototypical Networks**: Snell et al. "Prototypical Networks for Few-shot Learning." NeurIPS 2017
- **Meta-Learning**: Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017

### Frameworks
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [torchvision](https://pytorch.org/vision/stable/index.html) - Pre-trained models
- [scikit-learn](https://scikit-learn.org/) - Evaluation metrics

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: DeepFashion2 team for the comprehensive fashion dataset
- **Pre-trained Models**: PyTorch/torchvision for ResNet-50 weights
- **Community**: Open-source contributors to PyTorch, NumPy, and related libraries

---

## 📧 Contact

**Project Maintainer**: VISHAL-Nk

**Repository**: [Few-shot-Fashion-Category-Recognition-in-Emerging-Apparel-Markets](https://github.com/VISHAL-Nk/Few-shot-Fashion-Category-Recognition-in-Emerging-Apparel-Markets)

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

