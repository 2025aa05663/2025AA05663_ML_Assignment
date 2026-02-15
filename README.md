# ML Assignment 2 — Credit Card Default Prediction (UCI Dataset)

## Problem Statement:
The goal is to predict whether a customer will default on their credit card payment using different machine learning models and comparing their performance.

## Dataset Used

**UCI Default of Credit Card Clients Dataset**

- Rows: 30,000  
- Target column: "target"  
  - 0 → Not Default  
  - 1 → Default

The dataset is a binary classification problem with a bit class imbalance.

## Models Used
The following ML models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost

## Comparison Table:

| ML Model Name       | Accuracy           | AUC                | Precision           | Recall              | F1                  | MCC                 |
|---------------------|--------------------|--------------------|---------------------|---------------------|---------------------|---------------------|
| Logistic Regression | 0.8076666666666666 | 0.7076355036089734 | 0.6868250539956804  | 0.23963828183873398 | 0.3553072625698324  | 0.32444311464457076 |
| Decision Tree       | 0.7145             | 0.6074506323181914 | 0.36941813261163736 | 0.4114544084400904  | 0.3893048128342246  | 0.2042155615431836  |
| KNN                 | 0.8003333333333333 | 0.709969294013889  | 0.5819567979669632  | 0.34513941220798794 | 0.43330179754020814 | 0.33776609424548504 |
| Naive Bayes         | 0.7525             | 0.7249303386463403 | 0.4514742014742015  | 0.5538809344385832  | 0.49746192893401014 | 0.33862043928395225 |
| Random Forest       | 0.816              | 0.7727170032402468 | 0.6577086280056577  | 0.35041446872645066 | 0.45722713864306785 | 0.3844170606799151  |
| XGBoost             | 0.817              | 0.7780837374705112 | 0.6605890603085554  | 0.3549359457422758  | 0.46176470588235297 | 0.3888124775644108  |
  
## Observations




A Streamlit web application is also deployed to allow:
- uploading a test dataset (CSV file)
- selecting a trained model
- viewing predictions and evaluation metrics
- 
## Streamlit App (Live Demo)

https://2025aa05663mlassignment-klxjekmfa78grewjvovdje.streamlit.app


## How to Run Locally

1. Clone this repository:

git clone [https://github.com/2025aa05663/2025AA05663_ML_Assignment.git](https://github.com/2025aa05663/2025AA05663_ML_Assignment.git)

cd 2025AA05663_ML_Assignment

2. Install dependencies:
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run app.py

4. Open the link shown in terminal

