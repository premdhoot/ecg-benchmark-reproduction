# ECG Benchmark Reproduction

This repository contains a full reproduction of the paper:  
**"In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis"**  
by Naoki Nonaka and Jun Seita (ML4H, 2021)

## 📋 Overview

We reproduce the results for the ResNet18 architecture on the PTB-XL dataset to evaluate its performance for multi-label ECG classification. This includes data preprocessing, model implementation (1D ResNet18), training, evaluation, and a final report with reproducibility analysis.

## 📁 Structure

ecg-benchmark-reproduction/
├── data/ # PTB-XL raw and preprocessed data
├── models/ # ResNet18_1D model implementation
├── utils/ # Preprocessing and metrics scripts
├── train.py # Training loop
├── eval.py # Evaluation script
├── requirements.txt # Dependencies
└── report.pdf # Final written report

## 🚀 How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

2. **Download PTB-XL Data**

• Visit: https://physionet.org/content/ptb-xl/1.0.1/

• Download ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip

• Unzip into ./data/ptb-xl/

3. **Preprocess the Dataset**
    ```bash
    python utils/preprocess.py

4. **Train the Model**
    ```bash
    python train.py

5. **Evaluate the Model**
    ```bash
    python eval.py

## 📦 Dependencies
Python 3.10.12

PyTorch

Scikit-learn

Pandas, NumPy, Matplotlib

(Full list in requirements.txt)

## 📫 Author
Prem Dhoot
University of Illinois Urbana-Champaign
Email: premdhoot@illinois.edu

