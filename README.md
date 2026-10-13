# Multimodal Skin Lesion Classification Using Deep Learning

A deep learning project that classifies skin lesions using the **HAM10000 dataset**. This project compares image-based CNN models with multimodal models that combine dermoscopic images and patient metadata to predict seven lesion categories and classify lesions into two project-defined groups: benign and malignant.

## 🎯 Project Overview

The project evaluates four deep learning models to study whether combining patient metadata with image features can improve classification performance.

### Lesion Categories

The models classify lesions into seven categories:

| Label | Description |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

The binary classification experiment groups `nv`, `bkl`, `df`, and `vasc` as benign, and `mel`, `bcc`, and `akiec` as malignant for this project. This is a simplified experimental grouping, not a definitive clinical diagnosis.

## ✨ Features

- **Four-model comparison:** Evaluates EfficientNet-B0 and DenseNet121 using image-only and multimodal approaches.
- **Multimodal learning:** Combines dermoscopic image features with patient age and gender.
- **Transfer learning:** Uses CNN architectures pretrained on ImageNet.
- **Multitask classification:** Predicts both seven-class lesion categories and two-class labels.
- **Image preprocessing:** Resizes images to 224 × 224 pixels and applies ImageNet normalization.
- **Data augmentation:** Uses horizontal flips, vertical flips, and random rotations during training.
- **Lesion-level data splitting:** Separates lesions across training, validation, and test sets to reduce data leakage from repeated images of the same lesion.
- **Model evaluation:** Compares accuracy and F1-score for both classification tasks.

## 🧠 Models Compared

| Model | Input |
|---|---|
| EfficientNet-B0 | Dermoscopic image |
| DenseNet121 | Dermoscopic image |
| Multimodal EfficientNet-B0 | Image + age + gender |
| Multimodal DenseNet121 | Image + age + gender |

### Architecture Overview

**Image-only models**

Dermoscopic image → CNN backbone → Classification heads → Seven-class and binary predictions

**Multimodal models**

Dermoscopic image → CNN backbone ─┐  
Patient age and gender → Metadata network ─┴→ Feature fusion → Classification heads

The image branch extracts visual features, while the metadata branch processes patient information. The fused representation is used for both classification tasks.

## 📊 Experimental Results

The following table reports the measured results from the project experiments.

| Model | Binary Accuracy | Binary F1 | Lesion Accuracy | Lesion F1 |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 87.80% | 88.03% | 81.50% | 82.07% |
| DenseNet121 | 87.67% | 87.94% | 82.44% | 81.05% |
| Multimodal EfficientNet-B0 | 89.34% | 89.43% | 81.43% | 82.17% |
| Multimodal DenseNet121 | 88.47% | 88.75% | 84.12% | 84.57% |

### Key Observations

- **Best binary classification accuracy:** Multimodal EfficientNet-B0 achieved 89.34%.
- **Best seven-class lesion accuracy:** Multimodal DenseNet121 achieved 84.12%.
- **Best seven-class lesion F1-score:** Multimodal DenseNet121 achieved 84.57%.
- The results show that the impact of adding metadata differs between the two CNN architectures and classification tasks.

## 📁 Project Structure

```text
SkinCancer_project/
├── data/
│   └── metadata.csv
├── models/
├── notebooks/
│   └── skincancerproject.ipynb
├── results/
│   ├── model_comparison.csv
│   └── model_comparison.md
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── main.py
├── requirements.txt
├── test_dataset.py
├── test_model.py
├── .gitignore
└── README.md
```

The repository may contain additional source files. Check the implementation before using individual scripts. The dataset images and trained model checkpoints are not necessarily included in the repository.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible version for the installed dependencies
- PyTorch and Torchvision
- A CUDA-compatible GPU is recommended for training
- Sufficient disk space for the HAM10000 dataset

### 1. Clone the Repository

```bash
git clone https://github.com/Saireddy957/SkinCancer_project.git
cd SkinCancer_project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download HAM10000 from Kaggle:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Obtain the image files and metadata CSV. Arrange them according to the paths expected by the notebook or training scripts. The metadata must include the image identifier, lesion identifier, diagnosis label, age, and sex fields used by preprocessing.

**Note:** The repository's metadata file may be a placeholder. Verify that it contains the actual dataset records before running the experiments. The image files are not included in this repository by default.

## 📓 Experiments and Results

The `notebooks/` directory contains the research notebook used for image-based model experiments. The multimodal experiments are associated with the Kaggle notebook used during development.

The `results/` directory contains the model comparison results in CSV and Markdown formats.

For reproducibility, use the same preprocessing, lesion-level data split, label mapping, and evaluation procedure as the reported experiments.

## 🛠️ Technologies Used

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Models:** EfficientNet-B0, DenseNet121
- **Data Processing:** NumPy, Pandas
- **Evaluation:** Scikit-learn
- **Visualization:** Matplotlib and related plotting tools
- **Dataset:** HAM10000

## 📚 References

1. Tschandl, P., Rosendahl, C., and Kittler, H. “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.” *Scientific Data*, 5, 180161 (2018).
2. Esteva, A., et al. “Dermatologist-level classification of skin cancer with deep neural networks.” *Nature*, 542, 115–118 (2017).

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. Its predictions are not a substitute for professional medical evaluation, histopathology, or advice from a qualified healthcare professional. The binary labels used in this experiment are simplified research groupings and should not be interpreted as definitive clinical diagnoses.

## 👤 Author

**Sai Reddy**

GitHub: https://github.com/Saireddy957

---

If you find this project useful, consider starring the repository.
