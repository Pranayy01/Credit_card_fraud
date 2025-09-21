import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Credit Fraud Dashboard", layout="wide")

# --- simple styling
st.markdown("""
    <style>
    .big-font { font-size:22px !important; }
    .card { background:#0f1720; padding:14px; border-radius:10px; color: #e6eef8; }
    .muted { color:#9aa7b6; }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Fraud — Interactive Dashboard")
st.markdown("Upload a credit-card-like CSV (target column `Class` if available). Use the slider to tune threshold and inspect tradeoffs.")

# Sidebar
st.sidebar.header("Model & Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV (CSV)", type=["csv"])
model_path = st.sidebar.text_input("Keras model path", "shallow_nm.keras")
if st.sidebar.button("Load Model"):
    st.session_state['model_loaded'] = True

@st.cache_resource
def load_keras_model(path):
    try:
        return load_model(path)
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")
        return None

model = None
if 'model_loaded' in st.session_state or uploaded_file is not None:
    model = load_keras_model(model_path)
    if model:
        st.sidebar.success("Model loaded")

# Preprocess helper
def preprocess(df, target_col="Class"):
    if target_col in df.columns:
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
    else:
        X = df.copy()
        y = None
    # numeric only
    X = X.select_dtypes(include=[np.number]).fillna(df.median())
    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, y, feature_names, scaler

def compute_metrics(y_true, y_prob, thresh):
    y_pred = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = float("nan")
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "roc":roc}, y_pred

if uploaded_file is None:
    st.info("Upload a CSV to start. (Try the 'creditcard.csv' you uploaded earlier.)")
    st.stop()

# Read uploaded file
df = pd.read_csv(uploaded_file)
st.subheader("Data preview")
st.dataframe(df.head())

Xs, y, feature_names, scaler = preprocess(df, target_col="Class")
st.write(f"Detected **{len(feature_names)}** numeric features: {feature_names[:10]}{'...' if len(feature_names)>10 else ''}")

# train/val split if labels exist (for quick metrics)
if y is not None:
    X_tr, X_val, y_tr, y_val = train_test_split(Xs, y.values, test_size=0.2, random_state=42, stratify=y)
else:
    X_val, y_val = Xs, None

if model is None:
    st.error("Model not loaded. Check path in sidebar and press Load Model.")
    st.stop()

# predict probabilities on validation/all data
y_prob = model.predict(X_val).ravel()
default_thresh = 0.5

# Sidebar slider for threshold
st.sidebar.subheader("Decision Threshold")
thresh = st.sidebar.slider("Probability threshold", 0.0, 1.0, value=default_thresh, step=0.01)

# Layout: metrics on top
metrics, cm_col = st.columns([2,3])
with metrics:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='big-font'>Overall (Validation)</div>", unsafe_allow_html=True)
    if y_val is not None:
        m, y_pred = compute_metrics(y_val, y_prob, thresh)
        st.markdown(f"<div style='display:flex; gap:12px; margin-top:10px;'>"
                    f"<div><b>Accuracy</b><div class='muted'>{m['acc']:.4f}</div></div>"
                    f"<div><b>Precision</b><div class='muted'>{m['prec']:.4f}</div></div>"
                    f"<div><b>Recall</b><div class='muted'>{m['rec']:.4f}</div></div>"
                    f"<div><b>F1</b><div class='muted'>{m['f1']:.4f}</div></div>"
                    f"<div><b>ROC AUC</b><div class='muted'>{m['roc']:.4f}</div></div>"
                    f"</div>", unsafe_allow_html=True)
    else:
        st.markdown("No target column (`Class`) — model predictions shown for uploaded data.", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with cm_col:
    st.subheader("Confusion matrix (threshold: {:.2f})".format(thresh))
    if y_val is not None:
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.write("No true labels available.")

# ROC and PR curves
curve_col1, curve_col2 = st.columns(2)
with curve_col1:
    st.subheader("ROC Curve")
    if y_val is not None:
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, linewidth=2)
        ax.plot([0,1],[0,1], linestyle='--', alpha=0.6)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC AUC = {roc_auc:.4f}")
        st.pyplot(fig)
    else:
        st.write("No labels to compute ROC.")

with curve_col2:
    st.subheader("Precision-Recall Curve")
    if y_val is not None:
        prec, rec, _ = precision_recall_curve(y_val, y_prob)
        pr_auc = auc(rec, prec)
        fig, ax = plt.subplots()
        ax.plot(rec, prec, linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR AUC = {pr_auc:.4f}")
        st.pyplot(fig)
    else:
        st.write("No labels to compute PR curve.")

# Top-permutation importance (fast on subset)
if y_val is not None and st.checkbox("Show permutation feature importance (top 10, approximate)"):
    n_sample = min(3000, len(X_val))
    idxs = np.random.choice(len(X_val), n_sample, replace=False)
    try:
        with st.spinner("Computing permutation importance (this may take a moment)..."):
            r = permutation_importance(lambda X: model.predict(X).ravel(), X_val[idxs], y_val[idxs], n_repeats=6, random_state=42, scoring='roc_auc')
            imp = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False)[:10]
            st.bar_chart(imp)
    except Exception as e:
        st.error(f"Permutation importance failed: {e}")

# Per-row interactive predictor + download
st.subheader("Per-row predictions & download")
pred_df = pd.DataFrame({"probability": y_prob, "pred": (y_prob >= thresh).astype(int)})
display_df = pd.concat([pd.DataFrame(X_val, columns=feature_names).reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
st.dataframe(display_df.head(10))

# allow download
csv_bytes = display_df.to_csv(index=False).encode('utf-8')
st.download_button("Download predictions (CSV)", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("Tip: For production, persist the scaler and the exact feature order used during training; align uploaded data to that order before predicting.")
