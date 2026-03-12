# Deep Learning Practice 1: CNNs for STL-10 Classification

**Course:** Deep Learning (Master's degree in AI, 2025-2026).
**Authors:** Kai HERBST & Bellani Wassim.
**Institution:** Universidade da Coruña.

## 📌 Project Overview
This repository contains the code, models, and explanatory reports for Practice 1 of the Deep Learning course. The primary objective is to develop several Convolutional Neural Network (CNN) models to solve the image classification problem posed by the STL-10 dataset. 

The project is divided into two main phases:
1. **Custom CNNs:** Building architectures from scratch (Baseline, Wide, and Residual networks), implementing hyperparameter tuning, and applying regularization techniques (Dropout, Batch Normalization, L2/Ridge) to prevent overfitting.
2. **Pretrained CNNs:** Applying transfer learning (Feature Extraction and Fine-Tuning) using state-of-the-art models like ResNet50V2 and EfficientNetB0 from `keras.applications`.

## 📊 Dataset
The models are trained and evaluated on the **STL-10 dataset**.
* **Image Properties:** 96x96 pixels in color (RGB).
* **Data Split:** 5,000 labeled images for training and 8,000 labeled images for testing. 
* **Validation:** 20% of the training data (1,000 images) is automatically reserved for hyperparameter tuning to avoid data leakage.
* **Classes (10):** Airplane, bird, car, cat, deer, dog, horse, monkey, ship, and truck (perfectly balanced).

## 📁 Repository Structure
To maintain a clean and professional workflow, the machine learning pipeline is divided into a reusable data preparation script and numbered Jupyter Notebooks containing the experiments and explanatory reports.

* `requirements.txt`: Contains all the necessary Python dependencies and their specific versions to ensure reproducibility.
* `data_prep.py`: A modular Python script that downloads STL-10, resizes images, normalizes pixel values to [0, 1], applies one-hot encoding to labels, and handles the validation split.
* `01_Data_Preprocessing_and_Baseline.ipynb`: Initializes the data pipeline and establishes a baseline Sequential CNN model.
* `02_Custom_CNN_Architectures.ipynb`: Contains experiments with more complex, custom-built architectures (Wide CNNs, custom ResNet blocks) and regularization tuning.
* `03_Pretrained_Models.ipynb`: Implements transfer learning (Feature Extraction) using models pulled from Keras applications.
* `04_Final_Results_and_Comparison.ipynb`: Aggregates the results from all previous notebooks, providing a reasoned comparison of advantages, disadvantages, and a final summary table.

## ⚙️ Requirements & Execution

### 1. Install Dependencies
To ensure full compatibility and reproducibility, we have included a strict `requirements.txt` file. We recommend creating a virtual environment, then installing the required packages by running the following command in your terminal:

```bash
pip install -r requirements.txt
