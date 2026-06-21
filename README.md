# COEN807 Term Project: Breast Cancer Diagnosis Prediction

## Project Overview
This project applies supervised machine learning to predict whether a breast tumour is malignant or benign using the Breast Cancer Wisconsin Diagnostic dataset.

## Dataset
Dataset source: Breast Cancer Wisconsin Diagnostic dataset, available from UCI Machine Learning Repository and through scikit-learn.

Prepared dataset file:
`dataset/breast_cancer_wisconsin_diagnostic.csv`

Target column:
`diagnosis_malignant` where 1 = malignant and 0 = benign.

## Models Implemented
- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree
- Random Forest

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## How to Reproduce Results
1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the experiment:

```bash
python src/train_and_evaluate.py
```

4. Check generated files in the `outputs/` and `models/` folders.

## Folder Structure
```text
COEN807_Breast_Cancer_ML_Project/
├── dataset/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── slides/
├── src/
├── README.md
└── requirements.txt
```

## Best Model
The Support Vector Machine was selected as the best validation model. On the final test set it achieved 98.84% accuracy and 98.41% F1-score for malignant cancer detection.
