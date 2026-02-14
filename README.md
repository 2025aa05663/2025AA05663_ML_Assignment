# ML Assignment 2 — Credit Card Default Prediction (UCI Dataset)

This project was created as a part of ML Assignment. The goal is to build and compare multiple machine learning models for predicting whether a customer will default on their credit card payment.

A Streamlit web application is also deployed to allow:
- uploading a test dataset (CSV file)
- selecting a trained model
- viewing predictions and evaluation metrics

## Dataset Used

**UCI Default of Credit Card Clients Dataset**

- Rows: 30,000  
- Target column: "targe"  
  - 0 → Not Default  
  - 1 → Default

The dataset is a binary classification problem with a bit class imbalance.

---

## Models Implemented
The following ML models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost

All models were evaluated using:

- Accuracy  
- AUC  
- Precision  
- Recall  
- F1 Score  
- MCC  
- Confusion Matrix


## Repository Structure

ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── metrics_comparison.csv
└── model/
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── logistic_regression.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
