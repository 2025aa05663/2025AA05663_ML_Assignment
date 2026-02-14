import streamlit as st
import pandas as pd
import joblib
import os

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Page setup
st.set_page_config(page_title="ML Assignment - Credit Default", layout="wide")
st.title("ML Assignment : Credit Card Default Prediction")

# Model loading
MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

@st.cache_resource
def load_selected_model(model_key: str):
    model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_key])
    return joblib.load(model_path)

# Sidebar controls
st.sidebar.header("Inputs")
model_key = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
model = load_selected_model(model_key)

uploaded_file = st.sidebar.file_uploader("Upload CSV (test data)", type=["csv"])

# Helper functions
def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def confusion_matrix_figure(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    return fig

# Main logic
if uploaded_file is None:
    st.warning("Upload a CSV file to continue.")
    st.stop()

df_test = pd.read_csv(uploaded_file)

st.subheader("Uploaded Dataset Preview")
st.dataframe(df_test.head(10))

# If user uploads file with target column, compute metrics
if "target" in df_test.columns:
    y_true = df_test["target"]
    X = df_test.drop(columns=["target"])
    has_labels = True
else:
    X = df_test.copy()
    y_true = None
    has_labels = False

# Predict
y_pred = model.predict(X)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X)[:, 1]
else:
    y_prob = y_pred

# prediction output
out = X.copy()
out["Predicted_Default"] = y_pred
out["Default_Probability"] = y_prob

st.subheader("Predictions Output")
st.dataframe(out.head(15))

# Metrics + confusion matrix only if labels exist
if has_labels:
    st.subheader("Evaluation Metrics")
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    st.table(pd.DataFrame(metrics, index=["Value"]).T)

    st.subheader("Confusion Matrix")
    fig = confusion_matrix_figure(y_true, y_pred, f"Confusion Matrix - {model_key}")
    st.pyplot(fig)
else:
    st.info("No 'target' column found. Metrics and confusion matrix need true labels.")
