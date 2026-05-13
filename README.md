# 🩺 Skin Lesion Severity Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
Skin cancer is one of the most common cancers worldwide. Early detection through automated image analysis can significantly improve patient outcomes by triaging lesions before a dermatologist appointment. 

This project implements a Convolutional Neural Network (CNN) pipeline to classify dermoscopy images into two categories: **Benign** or **Malignant**. It features a robust data preprocessing pipeline, transfer learning using MobileNetV2, and an interactive web-based GUI for real-time inference.

This project was developed as part of the **Applied Machine Learning — EDGE Project** at the Dhaka University of Engineering & Technology (DUET).

---

## 📊 Dataset & Preprocessing

The model is trained on the [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), which contains over 10,000 dermoscopy images across 7 diagnostic categories. 

**Target Mapping:**
To create a binary classifier, the 7 classes were mapped as follows:
* **Malignant (Class 1):** Melanoma (`mel`), Basal cell carcinoma (`bcc`), Actinic keratoses (`akiec`).
* **Benign (Class 0):** Melanocytic nevi (`nv`), Benign keratosis-like (`bkl`), Dermatofibroma (`df`), Vascular (`vasc`).

**Handling Class Imbalance:**
Medical datasets are inherently imbalanced (heavily skewed toward benign). This was addressed by:
1.  **Data Augmentation:** Applying random rotations (20°), horizontal flips, and zooms (0.2) to the training set.
2.  **Class Weighting:** Computing balanced class weights dynamically during model training to heavily penalize misclassifications of the minority (malignant) class.

---

## 🧠 Model Architecture

Two distinct architectures were developed and evaluated:

1.  **Custom CNN:** A sequential model featuring 3 convolutional blocks (32, 64, 128 filters) with Max Pooling, followed by a dense layer with 50% Dropout.
2.  **Transfer Learning (MobileNetV2):** * **Base:** Pre-trained MobileNetV2 (frozen weights, trained on ImageNet).
    * **Head:** Custom `GlobalAveragePooling2D` layer followed by a `Dropout(0.5)` and a Sigmoid classification head.
    * **Result:** Significantly faster convergence and higher validation accuracy compared to the custom CNN.

---

## 📈 Evaluation Metrics

The models were rigorously evaluated on a hold-out test set using the following metrics:
* **Accuracy & Loss Curves:** To monitor overfitting.
* **Confusion Matrix:** To visualize True Positives (correctly identified malignancies) vs. False Negatives (missed diagnoses).
* **Classification Report:** Precision, Recall, and F1-Scores.
* **AUC-ROC Curve:** Measuring the model's capability to distinguish between classes across multiple probability thresholds.

---

## 💻 Interactive GUI

The project includes an interactive web application built with **Gradio**. 

**Features:**
* Upload any `.jpg` or `.png` dermoscopy image.
* Automatic resizing and normalization to $224 \times 224$ pixels.
* Real-time prediction output (Benign / Malignant).
* Confidence score visualizer.
* Actionable triage recommendations based on the prediction.



---

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/skin-lesion-classifier.git](https://github.com/yourusername/skin-lesion-classifier.git)
cd skin-lesion-classifier
