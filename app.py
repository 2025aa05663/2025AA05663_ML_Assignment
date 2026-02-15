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

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Credit Default Prediction",
    layout="wide"
)

st.title("ML Assignment : Credit Card Default Prediction")
st.write(
    "Select a trained model and upload a test CSV file to generate predictions and evaluate performance."
)

# ---------------------------
# Paths
# ---------------------------
MODEL_DIR = "model"
SAMPLE_DIR = "sample_data"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

SAMPLE_FILES = {
    "Sample CSV (with target)": "test_with_target.csv",
    "Sample CSV (without target)": "test_without_target.csv"
}

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model(model_name: str):
    model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    return joblib.load(model_path)

# ---------------------------
# Metrics + Confusion Matrix
# ---------------------------
def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

def confusion_matrix_figure(y_true, y_pred, title_text):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title_text)
    plt.tight_layout()
    return fig

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Inputs")

model_key = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
model = load_model(model_key)

uploaded_file = st.sidebar.file_uploader("Upload CSV (test data)", type=["csv"])

# ---------------------------
# Sample CSV Download Section
# ---------------------------
st.subheader("Download Sample Test Files")

col1, col2 = st.columns(2)

for i, (label, fname) in enumerate(SAMPLE_FILES.items()):
    sample_path = os.path.join(SAMPLE_DIR, fname)

    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            data = f.read()

        if i == 0:
            with col1:
                st.download_button(
                    label=label,
                    data=data,
                    file_name=fname,
                    mime="text/csv"
                )
        else:
            with col2:
                st.download_button(
                    label=label,
                    data=data,
                    file_name=fname,
                    mime="text/csv"
                )
    else:
        st.warning(f"Missing sample file: {sample_path}")

st.divider()

# ---------------------------
# Main Flow
# ---------------------------
if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to continue.")
    st.stop()

test_df = pd.read_csv(uploaded_file)

st.subheader("Uploaded Dataset Preview")
st.dataframe(test_df.head(10))

# Check target column
if "target" in test_df.columns:
    y_true = test_df["target"]
    X_test = test_df.drop(columns=["target"])
    labels_present = True
else:
    X_test = test_df.copy()
    y_true = None
    labels_present = False

# Predictions
y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
else:
    y_prob = y_pred

# Output predictions
out_df = X_test.copy()
out_df["Predicted_Default"] = y_pred
out_df["Default_Probability"] = y_prob

st.subheader("Predictions Output")
st.dataframe(out_df.head(20))

# Evaluation
if labels_present:
    st.subheader("Evaluation Metrics")

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics_table = pd.DataFrame(metrics, index=["Value"]).T
    st.table(metrics_table)

    st.subheader("Confusion Matrix")
    fig = confusion_matrix_figure(y_true, y_pred, f"Confusion Matrix - {model_key}")
    st.pyplot(fig)

else:
    st.warning("No 'target' column found. Metrics and confusion matrix require true labels.")
