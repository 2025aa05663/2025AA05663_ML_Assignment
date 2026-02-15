# ML Assignment 2 — Credit Card Default Prediction (UCI Dataset)

## Problem Statement:
The goal is to predict whether a customer will default on their credit card payment using different machine learning models and comparing their performance.

## Streamlit App (Live Demo)
https://2025aa05663mlassignment-klxjekmfa78grewjvovdje.streamlit.app

## GitHub Repository
[https://github.com/2025aa05663/2025AA05663_ML_Assignment.git](https://github.com/2025aa05663/2025AA05663_ML_Assignment.git)

## Dataset Used

**UCI Default of Credit Card Clients Dataset**

- Rows: 30,000  
- Target column: "target"  (The original target column was renamed to target for simplicity.)
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
| Logistic Regression | 0.8077 | 0.7076 | 0.6868  | 0.2396 | 0.3553  | 0.3244 |
| Decision Tree       | 0.7145 | 0.6075 | 0.3694  | 0.4114 | 0.3893  | 0.2042 |
| KNN                 | 0.8003 | 0.7100 | 0.5820  | 0.3451 | 0.4333  | 0.3378 |
| Naive Bayes         | 0.7525 | 0.7249 | 0.4515  | 0.5539 | 0.4975  | 0.3386 |
| Random Forest       | 0.8160 | 0.7727 | 0.6577  | 0.3504 | 0.4572  | 0.3844 |
| XGBoost             | 0.8170 | 0.7781 | 0.6606  | 0.3549 | 0.4618  | 0.3888 |
  
## Observations

| ML Model Name            | Observation about model performance                                                                                                                                                                                                         |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Achieved good accuracy (0.8077) but had very low recall (0.2396) indicating it predicts the majority class well (non-default), but it misses many actual defaulters.                                                                        |
| Decision Tree            | Performed the worst overall with lowest accuracy (0.7145), lowest AUC (0.6075), precision (0.3694) and F1 (0.3893) indicating poor generalization. MCC is the lowest (0.2042), confirming it is the weakest model, might be due to overfitting.             |
| kNN                      | High accuracy (0.8003), recall is moderate, meaning it still misses many defaulters. Better balance than Logistic Regression, but not the best overall.                                                                                     |
| Naive Bayes              | Highest recall (0.5539) catching more defaulters than any other model. Precision is lower (0.4515), meaning it produces more false positives. Best model if the priority is catching defaulters, but with more false alarms.                 |
| Random Forest (Ensemble) | Strong overall model with good ranking ability and balanced performance, high accuracy (0.8160) ,strong AUC (0.7727), MCC (0.3844) is high, indicating good balance between both classes.                                                   |
| XGBoost (Ensemble)       | Best overall results, with the highest accuracy (0.8170) and highest AUC (0.7781). It also has the best MCC (0.3888), showing the most balanced performance across both classes. Recall (0.3549) is moderate, but has good precision (0.6606) |


## Conclusion:
1. XGBoost performed best overall based on Accuracy, AUC, F1 and MCC.
2. Random Forest is a close second and also performs strongly.
3. Naive Bayes achieved the highest recall, making it suitable if catching defaulters is the    main objective.
4. Decision Tree performed worst and is not recommended as a final model.

## Sample Test Files

Two sample test CSV files are provided inside the `sample_data/` folder:

- `test_with_target.csv` → includes `target` column (shows evaluation metrics + confusion matrix)
- `test_without_target.csv` → does not include `target` column (only predictions)

These files can also be downloaded directly from the Streamlit app.
