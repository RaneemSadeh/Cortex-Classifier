# Cortex Classifier: A Deep Learning System for Brain Tumor Detection

Cortex Classifier is a deep learning-based medical imaging project focused on automated classification of brain tumors from MRI scans. It uses modern CNN architectures to classify tumors into four categories and supports radiologists with fast, accurate diagnoses.

---

## 1. Classification Objective

- **Input**: Brain MRI scans (sagittal, axial, coronal)
- **Output Classes**:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor

- **Dataset**: 5,249 labeled MRI images
- **Split**: 70% training, 15% validation, 15% testing

![Data Distribution](path/to/your/data_distribution_chart.png)

---

## 2. CNN Architectures & Methodology

### Architectures Compared

| Architecture     | Features                                           | Pros                                                   | Cons                            |
|------------------|----------------------------------------------------|--------------------------------------------------------|----------------------------------|
| AlexNet          | 8 layers, ReLU, GPU support                        | Simple baseline                                        | Overfitting, outdated            |
| ResNet-50        | 50 layers, skip connections                        | Handles deep learning without vanishing gradient       | High training time               |
| MobileNet-V2     | Inverted residual blocks                           | Efficient for edge/mobile deployment                   | Lower accuracy                   |
| EfficientNet-B0  | Compound scaling                                   | High accuracy + small size                             | Sensitive to hyperparameters     |

### Development Steps

- **Image Size**: 224x224
- **Normalization**: ImageNet standard
- **Tuning**: Learning rate, batch size, weight decay
- **Cross-validation**: 2-fold
- **Training Tool**: PyTorch + Adam optimizer
- **Hardware**: GPU-accelerated environment

---

## 3. Results & Evaluation

### Accuracy on Test Set

| Model           | Best Hyperparameters              | Accuracy     |
|----------------|-----------------------------------|--------------|
| EfficientNet-B0| LR: 0.001, WD: 0.001, BS: 32       | 97.84%       |
| MobileNet-V2   | LR: 0.001, WD: 0.001, BS: 32       | 97.21%       |
| AlexNet        | LR: 0.0001, WD: 0.001, BS: 32      | ~94%         |
| ResNet-50      | LR: 0.001, WD: 0.001, BS: 32       | 87.18%       |

### Confusion Matrices

![AlexNet Matrix](path/to/your/alexnet_matrix.png)
![ResNet-50 Matrix](path/to/your/resnet_matrix.png)
![MobileNet-V2 Matrix](path/to/your/mobilenet_matrix.png)
![EfficientNet-B0 Matrix](path/to/your/efficientnet_matrix.png)

### Learning Curves

![Accuracy Curve](path/to/your/accuracy_curve.png)
![Loss Curve](path/to/your/loss_curve.png)

---

## 4. User Interface

- **Features**:
  - Upload an MRI image
  - Select model (AlexNet, ResNet-50, etc.)
  - View prediction in real time

![UI Screenshot](path/to/your/interface.png)

---

## 5. Critical Evaluation & Future Plans

### Strengths

- High accuracy, especially EfficientNet-B0 and MobileNet-V2
- Reliable methodology with cross-validation
- Comparative analysis across multiple CNNs

### Areas for Improvement

- **Data Imbalance**: Use SMOTE or class weighting
- **Dataset Diversity**: Train on data from more MRI devices and regions
- **New Architectures**: Try Vision Transformers (ViTs)
- **Explainability**: Add Grad-CAM visualizations for model predictions
- **Deployment**: Support DICOM input and uncertainty estimation
