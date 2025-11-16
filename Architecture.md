# Architecture Documentation

## 🏛️ System Overview

This document provides a comprehensive architectural overview of the Few-Shot Learning system for Fashion Category Recognition, including detailed component diagrams and data flow specifications.

---

## 📐 Main Architecture Flow

```
┌─────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌─────────────────────┐      ┌──────────────────┐
│   Input     │─────▶│    ConvNeXt      │─────▶│   Part-Aware     │─────▶│  Meta-Learning      │─────▶│  Classification  │
│   Images    │      │    Feature       │      │    Pooling       │      │   Embedding         │      │     Layer        │
│  224×224×3  │      │   Extraction     │      │                  │      │    Network          │      │                  │
└─────────────┘      └──────────────────┘      └──────────────────┘      └─────────────────────┘      └──────────────────┘
                              │                          │                           │                          │
                              │                          │                           │                          │
                         768-dim                    768-dim                      512-dim                   13 classes
                         features                   fused                      embeddings                  (logits)
                                                   features
```

---

## 🔬 Detailed Component Architecture

### 1. Feature Extraction Module (ConvNeXt-Tiny)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         ConvNeXt-Tiny Backbone                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Input Image (224×224×3)                                                      │
│         │                                                                      │
│         ▼                                                                      │
│  ┌──────────────┐                                                             │
│  │ Stem Layer   │  Conv 4×4, stride=4                                         │
│  │   96 chs     │────────────────────────────────────────┐                    │
│  └──────────────┘                                        │                    │
│         │                                                 │                    │
│         ▼                                                 │                    │
│  ┌──────────────────────────────────────┐               │                    │
│  │  Stage 1 (3 blocks)                  │               │                    │
│  │  ┌────────────────────────┐          │               │                    │
│  │  │DepthwiseConv 7×7       │          │               │ Skip              │
│  │  │LayerNorm               │ ×3       │               │ Connections       │
│  │  │Linear (4× expansion)   │          │               │                    │
│  │  │GELU Activation         │          │               │                    │
│  │  │Linear (projection)     │──────────┤               │                    │
│  │  └────────────────────────┘          │               │                    │
│  │  96 channels                         │               │                    │
│  └──────────────────────────────────────┘               │                    │
│         │                                                 │                    │
│         ▼                                                 │                    │
│  ┌──────────────────────────────────────┐               │                    │
│  │  Downsampling Layer                  │               │                    │
│  │  LayerNorm + Conv 2×2, stride=2      │               │                    │
│  │  96 → 192 channels                   │               │                    │
│  └──────────────────────────────────────┘               │                    │
│         │                                                 │                    │
│         ▼                                                 │                    │
│  ┌──────────────────────────────────────┐               │                    │
│  │  Stage 2 (3 blocks)                  │               │                    │
│  │  ConvNeXt Blocks                     │               │                    │
│  │  192 channels                        │──────────────┤                    │
│  └──────────────────────────────────────┘               │                    │
│         │                                                 │                    │
│         ▼                                                 │                    │
│  ┌──────────────────────────────────────┐               │                    │
│  │  Downsampling Layer                  │               │                    │
│  │  192 → 384 channels                  │               │                    │
│  └──────────────────────────────────────┘               │                    │
│         │                                                 │                    │
│         ▼                                                 │                    │
│  ┌──────────────────────────────────────┐               │                    │
│  │  Stage 3 (9 blocks)                  │               │                    │
│  │  ConvNeXt Blocks                     │               │                    │
│  │  384 channels                        │──────────────┤                    │
│  └──────────────────────────────────────┘               │                    │
│         │                                                 │                    │
│         ▼                                                 │                    │
│  ┌──────────────────────────────────────┐               │                    │
│  │  Downsampling Layer                  │               │                    │
│  │  384 → 768 channels                  │               │                    │
│  └──────────────────────────────────────┘               │                    │
│         │                                                 │                    │
│         ▼                                                 │                    │
│  ┌──────────────────────────────────────┐               │                    │
│  │  Stage 4 (3 blocks)                  │               │                    │
│  │  ConvNeXt Blocks                     │               │                    │
│  │  768 channels                        │───────────────┘                    │
│  └──────────────────────────────────────┘                                    │
│         │                                                                      │
│         ▼                                                                      │
│  ┌──────────────────────────┐                                                │
│  │ Global Average Pooling   │                                                │
│  │   (7×7×768 → 768)        │                                                │
│  └──────────────────────────┘                                                │
│         │                                                                      │
│         ▼                                                                      │
│  Output: 768-dimensional feature vector                                      │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- **Parameters**: ~28M total, 0 trainable (frozen for feature extraction)
- **Input**: 224×224×3 RGB images (normalized with ImageNet stats)
- **Output**: 768-dimensional feature vectors
- **Design**: Hierarchical feature extraction with increasing receptive fields
- **Advantage**: 85%+ accuracy vs 54% with EfficientNet-B0

---

### 2. Part-Aware Pooling Module

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                       Part-Aware Pooling Architecture                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Feature Map (B×768×7×7)                  Segmentation Masks                  │
│         │                                         │                            │
│         │                                         │                            │
│         │                          ┌──────────────▼─────────────┐             │
│         │                          │  Segmentation Preprocessing │             │
│         │                          │  - Parse JSON polygons      │             │
│         │                          │  - Create full mask (224×224)│             │
│         │                          │  - Resize to 7×7           │             │
│         │                          └──────────────┬─────────────┘             │
│         │                                         │                            │
│         │                          ┌──────────────▼─────────────┐             │
│         │                          │  Horizontal Strip Division  │             │
│         │                          │  (5 parts)                 │             │
│         │                          │                             │             │
│         │                          │  Part 0: Rows 0-1  (top)   │             │
│         │                          │  Part 1: Rows 1-2          │             │
│         │                          │  Part 2: Rows 2-4  (middle)│             │
│         │                          │  Part 3: Rows 4-5          │             │
│         │                          │  Part 4: Rows 5-7  (bottom)│             │
│         │                          └──────────────┬─────────────┘             │
│         │                                         │                            │
│         │                                         ▼                            │
│         │                          Part Masks (B×5×7×7)                       │
│         │                                         │                            │
│         ▼                                         ▼                            │
│  ┌──────────────────────────────────────────────────────────────┐            │
│  │              Mask Normalization Layer                         │            │
│  │  For each part mask k:                                       │            │
│  │    mask_k_norm = (mask_k - min) / (max - min + ε)           │            │
│  │  Output: Normalized masks in [0, 1]                          │            │
│  └──────────────────────────────┬───────────────────────────────┘            │
│                                  │                                            │
│                                  ▼                                            │
│  ┌────────────────────────────────────────────────────────────────┐          │
│  │         Part-Aware Weighted Pooling (Per Part)                 │          │
│  │                                                                 │          │
│  │  For k = 0 to 4:                                               │          │
│  │    ┌─────────────────────────────────────────┐                │          │
│  │    │  mask_k: B×1×7×7                        │                │          │
│  │    │  feature_map: B×768×7×7                 │                │          │
│  │    │                                          │                │          │
│  │    │  weighted_features = mask_k * feature_map│                │          │
│  │    │                                          │                │          │
│  │    │  numerator = Σ(weighted_features)       │                │          │
│  │    │              over spatial dims (H×W)     │                │          │
│  │    │                                          │                │          │
│  │    │  denominator = Σ(mask_k) + ε            │                │          │
│  │    │                                          │                │          │
│  │    │  part_feature_k = numerator / denominator│               │          │
│  │    │  Output: B×768                          │                │          │
│  │    └─────────────────────────────────────────┘                │          │
│  │                                                                 │          │
│  └─────────────────────────────┬───────────────────────────────────┘          │
│                                 │                                            │
│                                 ▼                                            │
│                   Part Features (B×5×768)                                   │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │              Feature Fusion Layer                            │           │
│  │                                                               │           │
│  │  Option 1: Simple Average Pooling                            │           │
│  │    fused = mean(part_features, dim=1)                        │           │
│  │    Output: B×768                                             │           │
│  │                                                               │           │
│  │  Option 2: Attention-Based Fusion (Optional)                 │           │
│  │    attention_weights = softmax(learnable_params)             │           │
│  │    fused = Σ(attention_weights_k * part_features_k)          │           │
│  │    Output: B×768                                             │           │
│  └───────────────────────────────┬──────────────────────────────┘           │
│                                   │                                          │
│                                   ▼                                          │
│               Fused Features (B×768)                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

Legend:
  B = Batch size
  K = Number of parts (5)
  H×W = Spatial dimensions (7×7)
  C = Feature channels (768)
  ε = Small constant for numerical stability (1e-8)
```

**Key Operations:**
1. **Mask Generation**: Convert segmentation polygons → 7×7 spatial masks
2. **Part Division**: Split into 5 horizontal regions (head, upper, middle, lower, bottom)
3. **Weighted Pooling**: Spatial pooling guided by part masks
4. **Feature Fusion**: Aggregate part features into unified representation
5. **Output**: 768-dim fused feature for each sample

---

### 3. Meta-Learning Embedding Network

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    Embedding Network Architecture                              │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Input: Fused Features (B×768)                                                │
│         │                                                                      │
│         ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │                   Layer 1                                 │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ Linear(768 → 1024)     │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  │             │                                              │                │
│  │             ▼                                              │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ BatchNorm1d(1024)      │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  │             │                                              │                │
│  │             ▼                                              │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ ReLU (inplace=True)    │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  │             │                                              │                │
│  │             ▼                                              │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ Dropout(p=0.2)         │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  └─────────────┼──────────────────────────────────────────────┘                │
│                │                                                              │
│                ▼                                                              │
│         Output: B×1024                                                       │
│                │                                                              │
│                ▼                                                              │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │                   Layer 2                                 │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ Linear(1024 → 768)     │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  │             │                                              │                │
│  │             ▼                                              │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ BatchNorm1d(768)       │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  │             │                                              │                │
│  │             ▼                                              │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ ReLU (inplace=True)    │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  │             │                                              │                │
│  │             ▼                                              │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ Dropout(p=0.2)         │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  └─────────────┼──────────────────────────────────────────────┘                │
│                │                                                              │
│                ▼                                                              │
│         Output: B×768                                                        │
│                │                                                              │
│                ▼                                                              │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │                   Layer 3                                 │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ Linear(768 → 512)      │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  │             │                                              │                │
│  │             ▼                                              │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ BatchNorm1d(512)       │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  │             │                                              │                │
│  │             ▼                                              │                │
│  │  ┌────────────────────────┐                              │                │
│  │  │ ReLU (inplace=True)    │                              │                │
│  │  └──────────┬─────────────┘                              │                │
│  └─────────────┼──────────────────────────────────────────────┘                │
│                │                                                              │
│                ▼                                                              │
│         Unnormalized: B×512                                                  │
│                │                                                              │
│                ▼                                                              │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │          L2 Normalization (Optional)                     │                │
│  │  embedding_norm = embedding / ||embedding||_2            │                │
│  └──────────────────────────────┬───────────────────────────┘                │
│                                  │                                            │
│                                  ▼                                            │
│               Final Embeddings (B×512, unit normalized)                      │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

Network Properties:
  - Total Parameters: ~1.8M
  - Trainable Parameters: ~1.8M (during meta-learning)
  - Activation: ReLU with inplace operations
  - Regularization: Batch normalization + Dropout (0.2)
  - Output: 512-dim normalized embeddings in unit hypersphere
```

---

### 4. Prototypical Classification Layer

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    Prototypical Classifier Architecture                        │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────┐             │
│  │            Learnable Class Prototypes                        │             │
│  │  prototypes: Parameter(13×512)                               │             │
│  │  Randomly initialized, learned during training               │             │
│  └───────────────────────────┬──────────────────────────────────┘             │
│                               │                                                │
│  Query Embeddings (B×512)     │                                                │
│         │                     │                                                │
│         ▼                     ▼                                                │
│  ┌──────────────────────────────────────────────────────────────┐             │
│  │              Normalize Embeddings                            │             │
│  │  query_norm = query / ||query||_2                            │             │
│  │  proto_norm = prototypes / ||prototypes||_2                  │             │
│  └───────────────────────────┬──────────────────────────────────┘             │
│                               │                                                │
│                               ▼                                                │
│  ┌──────────────────────────────────────────────────────────────┐             │
│  │         Cosine Similarity Computation                        │             │
│  │                                                               │             │
│  │  similarities = query_norm @ proto_norm^T                    │             │
│  │  Shape: (B×512) @ (512×13) = B×13                            │             │
│  │                                                               │             │
│  │  For each sample i and class c:                              │             │
│  │    sim[i,c] = cos(query[i], prototype[c])                    │             │
│  │             = <query[i], prototype[c]> / (||q|| * ||p||)     │             │
│  │             = dot product (since both normalized)            │             │
│  └───────────────────────────┬──────────────────────────────────┘             │
│                               │                                                │
│                               ▼                                                │
│                    Similarities (B×13)                                         │
│                               │                                                │
│                               ▼                                                │
│  ┌──────────────────────────────────────────────────────────────┐             │
│  │           Temperature Scaling                                │             │
│  │                                                               │             │
│  │  temperature: Parameter(1), learnable                        │             │
│  │  Clamped to [0.01, 10.0] for stability                       │             │
│  │                                                               │             │
│  │  logits = similarities / clamp(temperature, 0.01, 10.0)      │             │
│  │                                                               │             │
│  │  Effect:                                                      │             │
│  │    - Low temp (→0.01): Sharp, confident predictions          │             │
│  │    - High temp (→10): Soft, uncertain predictions            │             │
│  └───────────────────────────┬──────────────────────────────────┘             │
│                               │                                                │
│                               ▼                                                │
│                    Final Logits (B×13)                                         │
│                               │                                                │
│                               ▼                                                │
│  ┌──────────────────────────────────────────────────────────────┐             │
│  │              Softmax (During Inference)                      │             │
│  │  probabilities = softmax(logits, dim=1)                      │             │
│  │  predicted_class = argmax(probabilities, dim=1)              │             │
│  └──────────────────────────────────────────────────────────────┘             │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

Classifier Properties:
  - Parameters: (13×512) + 1 = 6,657 parameters
  - All parameters are learnable
  - Distance Metric: Cosine similarity (equivalent to Euclidean on normalized vectors)
  - Temperature: Learnable scaling factor for calibration
  - Output: 13-way classification logits
```

---

## 🔄 Training Pipeline

### Phase 1: Meta-Learning (Episodic Training)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         Meta-Learning Episode                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Full Training Set (312,186 samples, 13 classes)                              │
│         │                                                                      │
│         ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │         Episode Construction (N-way K-shot)              │                │
│  │                                                           │                │
│  │  1. Randomly sample N=2 classes from 13 available        │                │
│  │  2. For each sampled class:                              │                │
│  │     - Sample K=13 support examples                       │                │
│  │     - Sample Q=15 query examples                         │                │
│  │                                                           │                │
│  │  Support Set: 2×13 = 26 examples                         │                │
│  │  Query Set: 2×15 = 30 examples                           │                │
│  └───────────────────────┬──────────────────────────────────┘                │
│                           │                                                    │
│                           ▼                                                    │
│  ┌────────────────────────────────────┬─────────────────────────────────┐    │
│  │      Support Set (26×768)          │    Query Set (30×768)           │    │
│  └────────────┬───────────────────────┴──────────┬──────────────────────┘    │
│               │                                   │                            │
│               ▼                                   ▼                            │
│  ┌────────────────────────┐        ┌────────────────────────┐                │
│  │  Embedding Network     │        │  Embedding Network     │                │
│  │  Support → 26×512      │        │  Query → 30×512        │                │
│  └────────────┬───────────┘        └────────────┬───────────┘                │
│               │                                   │                            │
│               ▼                                   │                            │
│  ┌────────────────────────────────┐              │                            │
│  │   Compute Class Prototypes     │              │                            │
│  │   For each of 2 classes:       │              │                            │
│  │   prototype[c] = mean of       │              │                            │
│  │   support embeddings for c     │              │                            │
│  │   Output: 2×512                │              │                            │
│  └────────────┬───────────────────┘              │                            │
│               │                                   │                            │
│               └───────────┬───────────────────────┘                            │
│                           │                                                    │
│                           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │          Distance Computation                            │                │
│  │  For each query and each prototype:                      │                │
│  │    dist[q,c] = ||query[q] - prototype[c]||²             │                │
│  │  Output: 30×2 distance matrix                            │                │
│  └───────────────────────┬──────────────────────────────────┘                │
│                           │                                                    │
│                           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │          Classification Logits                           │                │
│  │  logits = -distances                                     │                │
│  │  (negative distance = similarity)                        │                │
│  └───────────────────────┬──────────────────────────────────┘                │
│                           │                                                    │
│                           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │          Cross-Entropy Loss                              │                │
│  │  loss = CrossEntropy(logits, query_labels)               │                │
│  └───────────────────────┬──────────────────────────────────┘                │
│                           │                                                    │
│                           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │          Backpropagation & Update                        │                │
│  │  1. Compute gradients: ∇loss w.r.t. embedding params     │                │
│  │  2. Clip gradients (max_norm=1.0)                        │                │
│  │  3. Adam optimizer step (lr=0.001)                       │                │
│  │  4. Cosine annealing schedule step                       │                │
│  └──────────────────────────────────────────────────────────┘                │
│                                                                                │
│  Repeat for 2,000 episodes                                                   │
│  Validate every 100 episodes                                                 │
│  Save best model based on validation accuracy                                │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

Meta-Learning Hyperparameters:
  - Episodes: 2,000
  - N-way: 2 classes per episode
  - K-shot: 13 support examples per class
  - Q-query: 15 query examples per class
  - Optimizer: Adam (lr=0.001)
  - Scheduler: CosineAnnealing (T_max=2000)
  - Gradient Clipping: max_norm=1.0
  - Best Model: Saved based on validation accuracy
```

### Phase 2: Fine-Tuning on Full Dataset

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                      Fine-Tuning Pipeline                                      │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │         Initialize from Meta-Learned Weights             │                │
│  │  - Load best embedding network from Phase 1              │                │
│  │  - Create prototypical classifier (13 classes)           │                │
│  │  - Initialize prototypes from training data              │                │
│  └───────────────────────┬──────────────────────────────────┘                │
│                           │                                                    │
│                           ▼                                                    │
│  Full Training Set (312,186 samples, 13 classes)                              │
│         │                                                                      │
│         ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │         Create DataLoader (batch_size=256)               │                │
│  │  - Shuffle training data                                 │                │
│  │  - Use 4 workers for parallel loading                    │                │
│  │  - Pin memory for faster GPU transfer                    │                │
│  └───────────────────────┬──────────────────────────────────┘                │
│                           │                                                    │
│                           ▼                                                    │
│         For Each Epoch (25 total):                                            │
│                           │                                                    │
│                           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │                 Training Loop                                     │        │
│  │                                                                   │        │
│  │  For each batch (features, labels):                              │        │
│  │    │                                                              │        │
│  │    ├─▶ Forward Pass:                                             │        │
│  │    │   1. embeddings = embedding_net(features)  # B×512          │        │
│  │    │   2. logits = classifier(embeddings)       # B×13           │        │
│  │    │   3. loss = CrossEntropy(logits, labels)                    │        │
│  │    │                                                              │        │
│  │    ├─▶ Backward Pass:                                            │        │
│  │    │   1. optimizer.zero_grad()                                  │        │
│  │    │   2. loss.backward()                                        │        │
│  │    │   3. Clip gradients (max_norm=1.0) for both networks        │        │
│  │    │   4. optimizer.step()                                       │        │
│  │    │                                                              │        │
│  │    └─▶ Track Metrics:                                            │        │
│  │        - Accumulate loss                                         │        │
│  │        - Count correct predictions                               │        │
│  └───────────────────────┬──────────────────────────────────────────┘        │
│                           │                                                    │
│                           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │              Validation Loop                                      │        │
│  │                                                                   │        │
│  │  For each validation batch:                                      │        │
│  │    1. embeddings = embedding_net(features)                       │        │
│  │    2. logits = classifier(embeddings)                            │        │
│  │    3. Collect predictions and labels                             │        │
│  │                                                                   │        │
│  │  Compute Metrics:                                                │        │
│  │    - Accuracy: correct/total                                     │        │
│  │    - Precision, Recall, F1: sklearn metrics                      │        │
│  │    - Per-class F1 scores                                         │        │
│  └───────────────────────┬──────────────────────────────────────────┘        │
│                           │                                                    │
│                           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │           Learning Rate Scheduling                       │                │
│  │  ReduceLROnPlateau:                                      │                │
│  │    - Monitor: Validation F1                              │                │
│  │    - Patience: 3 epochs                                  │                │
│  │    - Factor: 0.5 (halve LR)                              │                │
│  └───────────────────────┬──────────────────────────────────┘                │
│                           │                                                    │
│                           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │           Model Checkpointing                            │                │
│  │  If validation_f1 > best_f1:                             │                │
│  │    - Save embedding_net state_dict                       │                │
│  │    - Save classifier state_dict                          │                │
│  │    - Save epoch and best_f1                              │                │
│  │    - Filename: optimal_fewshot_model.pt                  │                │
│  └──────────────────────────────────────────────────────────┘                │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

Fine-Tuning Hyperparameters:
  - Epochs: 25
  - Batch Size: 256
  - Optimizer: Adam with differential learning rates
    * Embedding Network: lr=0.0001
    * Classifier: lr=0.001
  - Weight Decay: 0.0001
  - Scheduler: ReduceLROnPlateau (monitor=F1, patience=3, factor=0.5)
  - Gradient Clipping: max_norm=1.0
  - Best Model: Saved based on validation F1-score
```

---

## 📊 Data Flow Dimensions

### Complete Pipeline Data Transformation

```
Stage 0: Raw Input
  └─▶ Images: 224×224×3 RGB (uint8, [0, 255])
  
Stage 1: Preprocessing
  └─▶ Normalized: 224×224×3 (float32, ImageNet normalized)
  
Stage 2: ConvNeXt Feature Extraction
  └─▶ Features: 768-dim (float32)
  
Stage 3: Spatial Feature Map (for PAP)
  └─▶ Spatial: 768×7×7 (float32, reshaped from global features)
  
Stage 4: Part-Aware Pooling
  ├─▶ Part Features: 5×768 (float32, per-part representations)
  └─▶ Fused Features: 768-dim (float32, aggregated)
  
Stage 5: Embedding Network
  └─▶ Embeddings: 512-dim (float32, L2-normalized)
  
Stage 6: Prototypical Classifier
  ├─▶ Logits: 13-dim (float32, raw scores)
  └─▶ Probabilities: 13-dim (float32, softmax normalized)
  
Stage 7: Prediction
  └─▶ Class Label: scalar (int, range [0, 12])
```

### Memory Footprint (Single Sample)

| Stage | Output Shape | Data Type | Memory |
|-------|--------------|-----------|--------|
| Raw Image | 224×224×3 | uint8 | 150 KB |
| Normalized | 224×224×3 | float32 | 602 KB |
| ConvNeXt Features | 768 | float32 | 3 KB |
| Spatial Features | 768×7×7 | float32 | 150 KB |
| Part Features | 5×768 | float32 | 15 KB |
| Fused Features | 768 | float32 | 3 KB |
| Embeddings | 512 | float32 | 2 KB |
| Logits | 13 | float32 | 52 B |

**Total per sample**: ~925 KB (peak during processing)

### Batch Processing (Batch Size = 256)

| Stage | Batch Shape | Memory |
|-------|-------------|--------|
| Input Batch | 256×224×224×3 | 38.4 MB |
| Feature Batch | 256×768 | 0.79 MB |
| Embedding Batch | 256×512 | 0.52 MB |
| Logits Batch | 256×13 | 13.3 KB |

**GPU Memory Usage** (approx):
- Model Parameters: ~120 MB (ConvNeXt + Embedding + Classifier)
- Activations (batch=256): ~200 MB
- Gradients (training): ~120 MB
- **Total**: ~440 MB (comfortable on 4+ GB GPUs)

---

## 🎯 Design Decisions & Rationale

### 1. Why ConvNeXt over EfficientNet?

| Aspect | EfficientNet-B0 | ConvNeXt-Tiny |
|--------|-----------------|---------------|
| Accuracy | 54% | **85%+** |
| Parameters | 5.3M | 28M |
| Feature Dim | 1280 | 768 |
| Training Speed | Faster | Moderate |
| Generalization | Good | **Excellent** |

**Decision**: ConvNeXt for 31% accuracy improvement

### 2. Why Part-Aware Pooling?

- **Fashion-Specific**: Different regions (collar, sleeves, hem) are discriminative
- **Spatial Attention**: Focuses on relevant parts, ignores background
- **Performance Gain**: ~3-5% accuracy improvement over global pooling
- **Interpretability**: Can visualize which parts contribute to predictions

### 3. Why Prototypical Networks?

- **Few-Shot Native**: Designed for limited data per class
- **Metric Learning**: Learns meaningful distance metric in embedding space
- **Generalizable**: Works well even with new unseen classes
- **Simple & Effective**: No complex meta-learner, just distance-based classification

### 4. Two-Phase Training Strategy

**Phase 1 (Meta-Learning)**:
- Learn general-purpose embedding space
- Fast adaptation to new tasks
- Prevents overfitting to specific classes

**Phase 2 (Fine-Tuning)**:
- Optimize for specific 13-class problem
- Leverages full dataset
- Achieves higher accuracy on target distribution

---

## 🔧 Implementation Notes

### Optimization Techniques

1. **Batch-wise Processing**
   - Prevents OOM errors on large datasets
   - Memory-mapped arrays for feature storage
   - Aggressive garbage collection

2. **Mixed Precision Training** (Optional)
   - Use `torch.cuda.amp` for 2× speedup
   - Maintain numerical stability with loss scaling

3. **Gradient Clipping**
   - Prevents exploding gradients in meta-learning
   - Max norm = 1.0 for stability

4. **Data Loading**
   - 4 parallel workers for I/O
   - Pin memory for faster GPU transfer
   - Prefetch batches during computation

### Reproducibility

- Fixed random seeds (42) for all libraries
- Deterministic CUDA operations
- Saved model checkpoints include full state

### Extensibility

- **New Backbones**: Easy to swap ConvNeXt with other models
- **More Parts**: Configurable number of spatial regions
- **Different Tasks**: Can adapt to object detection, segmentation
- **Transfer Learning**: Pre-trained embeddings for related domains

---

## 📈 Performance Characteristics

### Training Time (NVIDIA GPU)

| Phase | Duration | GPU Memory |
|-------|----------|------------|
| Preprocessing | ~30 min | 2 GB |
| Feature Extraction | ~45 min | 4 GB |
| PAP Processing | ~20 min | 6 GB |
| Meta-Learning | ~2 hours | 8 GB |
| Fine-Tuning | ~3 hours | 8 GB |
| **Total** | **~6.5 hours** | **8 GB** |

### Inference Speed

- **Single Sample**: ~15 ms (CPU), ~2 ms (GPU)
- **Batch (256)**: ~3 sec (CPU), ~0.5 sec (GPU)
- **Throughput**: ~500 samples/sec (GPU)

### Scalability

- **Dataset Size**: Tested up to 500K samples
- **Batch Size**: Up to 512 on 16GB GPU
- **Number of Classes**: Scalable to 50+ classes
- **Image Resolution**: 224×224 (can adjust to 384×384 for higher quality)

---

## 🔍 Future Improvements

1. **Attention Mechanisms**
   - Self-attention in part fusion
   - Cross-attention between support and query sets

2. **Advanced Meta-Learning**
   - MAML (Model-Agnostic Meta-Learning)
   - Matching Networks with full context embeddings

3. **Multi-Task Learning**
   - Joint training on classification + attribute prediction
   - Style transfer auxiliary task

4. **Data Augmentation**
   - Advanced augmentations (CutMix, MixUp)
   - Synthetic data generation with GANs

5. **Model Compression**
   - Knowledge distillation to smaller models
   - Quantization for mobile deployment

---

This architecture documentation provides a comprehensive blueprint for understanding and extending the Few-Shot Fashion Category Recognition system. For implementation details, refer to `CompletePipeLine.ipynb`.
