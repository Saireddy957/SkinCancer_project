# Skin Cancer Detection using Multimodal Deep Learning

A deep learning project for detecting and classifying skin cancer lesions using the HAM10000 dataset. This project implements a **multimodal CNN** that combines image features with patient metadata (age, sex, localization) for improved classification accuracy.

## 🎯 Project Overview

This project classifies skin lesions into 7 categories:
- **akiec**: Actinic keratoses and intraepithelial carcinoma
- **bcc**: Basal cell carcinoma
- **bkl**: Benign keratosis-like lesions
- **df**: Dermatofibroma
- **mel**: Melanoma
- **nv**: Melanocytic nevi
- **vasc**: Vascular lesions

## ✨ Features

- **Multimodal Architecture**: Combines CNN image features with patient metadata
- **Multiple Backbone Support**: ResNet50, EfficientNet-B0, MobileNet-V2
- **Bio-inspired Optimization**: Ant Colony Optimization (ACO) for hyperparameter tuning
- **Comprehensive Evaluation**: Confusion matrix, ROC curves, classification reports
- **Data Augmentation**: Random flips, rotations, color jitter
- **Transfer Learning**: Pretrained ImageNet weights

## 📁 Project Structure

```
SkinCancer_Detection/
│
├── data/
│   ├── images/              # HAM10000 images
│   └── metadata.csv         # HAM10000 metadata file
│
├── notebooks/               # Jupyter notebooks for experiments
│
├── src/
│   ├── dataset.py           # Custom Dataset class
│   ├── model.py             # Multimodal CNN model
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── aco.py               # Ant Colony Optimization
│   └── utils.py             # Helper functions
│
├── models/                  # Saved trained models
│
├── results/                 # Evaluation results and plots
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── main.py                  # Main entry point
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository** (or download the project):
   ```bash
   cd SkinCancer_Detection
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup

1. **Download the HAM10000 dataset**:
   - Dataset: [Kaggle - HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
   - You need both image parts (HAM10000_images_part_1 and HAM10000_images_part_2)
   - And the metadata CSV file

2. **Place the data**:
   - Extract all images to `data/images/`
   - Place `HAM10000_metadata.csv` as `data/metadata.csv`

## 📊 Usage

### Training

**Option 1: Using the main entry point**
```bash
python main.py --mode train
```

**Option 2: Direct training script**
```bash
cd src
python train.py
```

**Custom training configuration**:
```python
config = {
    'data_dir': '../data/images',
    'metadata_file': '../data/metadata.csv',
    'model_type': 'multimodal',  # or 'image_only'
    'backbone': 'resnet50',       # or 'efficientnet_b0', 'mobilenet_v2'
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 0.001
}
```

### Evaluation

```bash
python main.py --mode evaluate --model_path models/best_model.pth
```

Or directly:
```bash
cd src
python evaluate.py
```

### Hyperparameter Optimization with ACO

```bash
python main.py --mode optimize
```

Or:
```bash
cd src
python aco.py
```

## 🏗️ Model Architecture

### Multimodal CNN

```
Input Image (224x224x3) ──┐
                          │
                          ├──> CNN Backbone ──> Image Features (2048-dim)
                          │    (ResNet50)                   │
                          │                                 │
Metadata (age, sex) ──────┤                                 ├──> Fusion ──> Classifier ──> 7 Classes
                          │                                 │      Network
                          └─────> MLP ──> Metadata Features (16-dim)
```

### Key Components

1. **Image Branch**: 
   - Pretrained CNN (ResNet50/EfficientNet/MobileNet)
   - Extracts visual features from dermoscopic images

2. **Metadata Branch**:
   - Multi-layer perceptron (MLP)
   - Processes patient metadata (age, sex, localization)

3. **Fusion Layer**:
   - Concatenates image and metadata features
   - Final classification through fully connected layers

## 📈 Results

Expected performance metrics on HAM10000:

| Metric | Score |
|--------|-------|
| Accuracy | ~85-90% |
| Weighted F1 | ~0.85 |
| AUC-ROC | ~0.90+ |

*Note: Results may vary based on hyperparameters and data split*

## 🔧 Key Files Description

- **`dataset.py`**: Custom PyTorch Dataset for loading images and metadata
- **`model.py`**: Neural network architectures (multimodal and image-only)
- **`train.py`**: Training loop with validation
- **`evaluate.py`**: Model evaluation with metrics and visualizations
- **`aco.py`**: Ant Colony Optimization for hyperparameter tuning
- **`utils.py`**: Utility functions (checkpointing, plotting, etc.)
- **`main.py`**: Unified entry point for all operations

## 🧪 Experiments

The `notebooks/` directory can be used for:
- Data exploration and visualization
- Model prototyping
- Results analysis
- Custom experiments

## 📚 References

1. HAM10000 Dataset:
   - Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018).

2. Deep Learning for Skin Cancer:
   - Esteva, A., et al. "Dermatologist-level classification of skin cancer with deep neural networks." Nature 542.7639 (2017): 115-118.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is for educational purposes. Please cite the HAM10000 dataset if you use it in your research.

## 🙏 Acknowledgments

- HAM10000 dataset creators
- PyTorch team
- Open-source deep learning community

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Happy Coding! 🎉**
