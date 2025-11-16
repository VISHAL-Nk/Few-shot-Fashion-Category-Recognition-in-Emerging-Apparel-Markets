# Few-Shot Learning for Fashion Category Recognition

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project implements a state-of-the-art **Few-Shot Learning** system for fashion apparel category recognition using the DeepFashion2 dataset. The system leverages advanced deep learning techniques including ConvNeXt feature extraction, Part-Aware Pooling, and Prototypical Meta-Learning to achieve robust classification even with limited training examples per category.

## 🎯 Key Features

- **Advanced Feature Extraction**: Utilizes ConvNeXt-Tiny (768-dim features) for superior performance over traditional architectures
- **Part-Aware Pooling**: Spatial attention mechanism that focuses on discriminative regions of fashion items
- **Meta-Learning Framework**: Prototypical networks with episodic training for few-shot generalization
- **Comprehensive Pipeline**: End-to-end system from data preprocessing to model evaluation
- **High Performance**: Achieves 85%+ accuracy with optimized embedding networks

## 🏗️ Architecture

The pipeline consists of four main stages:

```
Input Images → ConvNeXt Feature Extraction → Part-Aware Pooling → Meta-Learning Network → Category Classification
```

For detailed architecture diagrams and component descriptions, see [Architecture.md](Architecture.md).

## 📊 Dataset

**DeepFashion2**: A large-scale fashion dataset containing:
- **Training Set**: 312,186 samples
- **Validation Set**: 52,490 samples
- **Test Set**: Variable size
- **Categories**: 13 fashion categories
  - Short/Long sleeve tops and outwear
  - Vest, Sling, Shorts, Trousers, Skirt
  - Short/Long sleeve dresses, Vest dress, Sling dress

Each sample includes:
- High-resolution fashion images
- Bounding box annotations
- Segmentation masks (polygons)
- Landmark keypoints
- Category and style labels

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch >= 2.0
- torchvision
- numpy
- pandas
- opencv-python
- scikit-learn
- matplotlib
- seaborn
- tqdm

### Running the Pipeline

The complete pipeline is implemented in `CompletePipeLine.ipynb`. Execute the notebook cells sequentially:

1. **Data Preprocessing**: Image resizing and annotation normalization
2. **Feature Extraction**: ConvNeXt-based feature extraction
3. **Part-Aware Pooling**: Spatial attention and feature fusion
4. **Meta-Learning**: Episodic training and fine-tuning
5. **Evaluation**: Comprehensive performance metrics

## 📈 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | 85%+ |
| **F1-Score (Weighted)** | 0.84+ |
| **Precision** | 0.85+ |
| **Recall** | 0.84+ |

### Training Configuration

**Meta-Learning Phase:**
- Episodes: 2,000
- N-way: 2
- K-shot: 13
- Query samples: 15
- Learning rate: 0.001

**Fine-Tuning Phase:**
- Epochs: 25
- Batch size: 256
- Embedding LR: 0.0001
- Classifier LR: 0.001

## 🔑 Key Components

### 1. Feature Extraction
- **Backbone**: ConvNeXt-Tiny (pretrained on ImageNet)
- **Output**: 768-dimensional feature vectors
- **Advantages**: Better than EfficientNet-B0 (54% → 85% accuracy)

### 2. Part-Aware Pooling
- Divides features into 5 horizontal regions
- Applies weighted spatial pooling per part
- Fuses part features with attention mechanism
- Output: 768-dim fused features

### 3. Meta-Learning
- **Embedding Network**: 768 → 512 dimensions with skip connections
- **Prototypical Classifier**: Learnable class prototypes with temperature scaling
- **Training**: Episodic meta-learning followed by full dataset fine-tuning

## 📁 Project Structure

```
Project/
├── CompletePipeLine.ipynb          # Main notebook with complete pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Architecture.md                 # Detailed architecture documentation
├── DeepFashion2/                   # Dataset directory
│   ├── deepfashion2_original_images/
│   └── img_info_dataframes/
└── output/                         # Generated outputs
    ├── input/                      # Preprocessed data
    ├── resized/                    # Resized images
    ├── FEATURES_DIR/               # Extracted features
    ├── PartAwarePooling/           # PAP features
    ├── models/                     # Trained models
    │   ├── optimal_embedding_net.pt
    │   └── optimal_fewshot_model.pt
    └── results/                    # Evaluation results
        ├── optimal_fewshot_results.json
        ├── optimal_fewshot_training_curves.png
        └── optimal_fewshot_confusion_matrix.png
```

## 🔬 Technical Details

### Data Preprocessing
- **Resizing**: All images normalized to 224×224 pixels
- **Normalization**: ImageNet mean/std (transfer learning)
- **Augmentation**: Random crop, flip, rotation, color jitter (training only)
- **Annotations**: Bounding boxes and segmentation masks rescaled accordingly

### Memory Optimization
- Batch-wise feature processing to prevent OOM errors
- Memory-mapped arrays for large feature matrices
- Aggressive garbage collection and cache clearing
- Efficient numpy/torch tensor handling

### GPU Utilization
- CUDA-enabled training when available
- Automatic device detection and allocation
- Pinned memory for faster GPU transfers
- Mixed precision training support

## 📊 Model Outputs

The trained system produces:
- **Embedding Network**: Learned 512-dim embedding space
- **Class Prototypes**: 13 category prototypes in embedding space
- **Predictions**: Category labels with confidence scores
- **Metrics**: Accuracy, Precision, Recall, F1-score per category
- **Visualizations**: Training curves, confusion matrix, attention maps

## 🎓 Applications

- **Fashion E-commerce**: Automated product categorization
- **Style Recommendation**: Similar item retrieval
- **Visual Search**: Find products from images
- **Quality Control**: Detect mislabeled products
- **Trend Analysis**: Track fashion category popularity

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📚 References

1. **DeepFashion2**: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images
2. **Prototypical Networks**: Prototypical Networks for Few-shot Learning (Snell et al., 2017)
3. **ConvNeXt**: A ConvNet for the 2020s (Liu et al., 2022)
4. **Part-Based Models**: Part-based R-CNNs for Fine-grained Category Detection

## 👥 Authors

- VISHAL-Nk

## 🙏 Acknowledgments

- DeepFashion2 dataset creators
- PyTorch and torchvision teams
- Open-source community for various tools and libraries

---

**Note**: This project is part of academic research on few-shot learning applications in fashion domain.
