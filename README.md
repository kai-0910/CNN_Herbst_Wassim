# Deep Learning Practice 1: CNNs for STL-10 Classification

[cite_start]**Course:** Deep Learning (Master's degree in AI, 2025-2026) [cite: 5, 6]
**Authors:** Kai HERBST & Bellani Wassim.
[cite_start]**Institution:** Universidade da Coruña [cite: 2]

## 📌 Project Overview
[cite_start]This repository contains the code, models, and explanatory reports for Practice 1 of the Deep Learning course[cite: 5, 6]. [cite_start]The primary objective is to develop several Convolutional Neural Network (CNN) models to solve the image classification problem posed by the STL-10 dataset[cite: 11]. 

[cite_start]The project is divided into two main phases[cite: 17, 32]:
1. [cite_start]**Custom CNNs:** Building architectures from scratch (Baseline, Wide, and Residual networks), implementing hyperparameter tuning, and applying regularization techniques (Dropout, Batch Normalization, L2/Ridge) to prevent overfitting[cite: 22, 23, 24, 25, 26].
2. [cite_start]**Pretrained CNNs:** Applying transfer learning (Feature Extraction and Fine-Tuning) using state-of-the-art models like ResNet50V2 and EfficientNetB0 from `keras.applications`[cite: 33, 34, 35, 36].

## 📊 Dataset
[cite_start]The models are trained and evaluated on the **STL-10 dataset**[cite: 11, 13].
* [cite_start]**Image Properties:** 96x96 pixels in color (RGB)[cite: 13].
* [cite_start]**Data Split:** 5,000 labeled images for training and 8,000 labeled images for testing[cite: 13]. 
* [cite_start]**Validation:** 20% of the training data (1,000 images) is automatically reserved for hyperparameter tuning to avoid data leakage[cite: 21].
* [cite_start]**Classes (10):** Airplane, bird, car, cat, deer, dog, horse, monkey, ship, and truck (perfectly balanced)[cite: 15].

## 📁 Repository Structure
[cite_start]To maintain a clean and professional workflow, the machine learning pipeline is divided into a reusable data preparation script and numbered Jupyter Notebooks containing the experiments and explanatory reports[cite: 48, 49, 50, 55].

* [cite_start]`data_prep.py`: A modular Python script that downloads STL-10, resizes images, normalizes pixel values to [0, 1], applies one-hot encoding to labels, and handles the validation split[cite: 19, 20, 21].
* [cite_start]`01_Data_Preprocessing_and_Baseline.ipynb`: Initializes the data pipeline and establishes a baseline Sequential CNN model[cite: 49, 50, 53].
* [cite_start]`02_Custom_CNN_Architectures.ipynb`: Contains experiments with more complex, custom-built architectures (Wide CNNs, custom ResNet blocks) and regularization tuning[cite: 22, 26, 49, 50].
* [cite_start]`03_Pretrained_Models.ipynb`: Implements transfer learning (Feature Extraction) using models pulled from Keras applications[cite: 35, 49, 50].
* [cite_start]`04_Final_Results_and_Comparison.ipynb`: Aggregates the results from all previous notebooks, providing a reasoned comparison of advantages, disadvantages, and a final summary table[cite: 27, 29, 31, 37, 39, 41].

## ⚙️ Requirements & Execution
This project utilizes the following core libraries:
* `tensorflow` & `keras` (Model building and training)
* `tensorflow_datasets` (Data fetching)
* `matplotlib` (Evaluation and plotting)
* `pandas` (Result aggregation)

**How to Run:**
Ensure all dependencies are installed. Keep `data_prep.py` in the root directory alongside the notebooks. Run the notebooks in numerical order from `01` to `04` to follow the logical progression of the experiments. [cite_start]All cells must be executed top-to-bottom to reproduce the machine learning processes and final accuracy metrics[cite: 53, 54].
