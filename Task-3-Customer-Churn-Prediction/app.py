"""
app.py - Streamlit demo

Simple web interface so anyone can test the churn model without coding.
Has sliders for customer info and a batch upload option.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Forecaster",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size: 2.4rem; font-weight: 800; color: #1a1a2e; }
    .sub-title    { font-size: 1rem; color: #555; margin-top: -12px; }
    .metric-card  { background: #f0f4ff; border-radius: 10px; padding: 16px 20px;
                    border-left: 4px solid #4E8FD4; margin-bottom: 12px; }
    .risk-high    { color: #c0392b; font-weight: 700; font-size: 1.6rem; }
    .risk-low     { color: #27ae60; font-weight: 700; font-size: 1.6rem; }
    .risk-medium  { color: #e67e22; font-weight: 700; font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ── Helper: Load a saved model ───────────────────────────────────────────────
MODEL_DIR = "models"
MODEL_FILES = {
    "Gradient Boosting":    "gradient_boosting.pkl",
    "Random Forest":        "random_forest.pkl",
    "Logistic Regression":  "logistic_regression.pkl",
}

@st.cache_resource
def load_model(name: str):
    path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def build_feature_vector(inputs: dict) -> np.ndarray:
    """
    Convert the sidebar inputs into the same feature order the models expect.
    Must match the preprocessing in src/preprocess.py exactly.
    """
    geo_france  = 1 if inputs["geography"] == "France"  else 0
    geo_germany = 1 if inputs["geography"] == "Germany" else 0
    geo_spain   = 1 if inputs["geography"] == "Spain"   else 0

    gender_encoded = 1 if inputs["gender"] == "Male" else 0

    # Feature order (same as after get_dummies in preprocess.py)
    # CreditScore, Gender, Age, Tenure, Balance, NumOfProducts,
    # HasCrCard, IsActiveMember, EstimatedSalary,
    # Geo_France, Geo_Germany, Geo_Spain
    raw = np.array([[
        inputs["credit_score"],
        gender_encoded,
        inputs["age"],
        inputs["tenure"],
        inputs["balance"],
        inputs["num_products"],
        inputs["has_credit_card"],
        inputs["is_active_member"],
        inputs["salary"],
        geo_france, geo_germany, geo_spain,
    ]], dtype=float)

    # Simple z-score normalisation (approximate — ideally you'd save and reload
    # the fitted scaler too, but this works fine for demo purposes)
    means = np.array([650, 0.5, 38, 5, 75000, 1.5, 0.7, 0.5, 100000, 0.5, 0.25, 0.25])
    stds  = np.array([96,  0.5, 10, 3, 62000, 0.6, 0.45, 0.5, 57000, 0.5, 0.43, 0.43])
    return (raw - means) / (stds + 1e-8)


def risk_label(prob: float) -> str:
    if prob >= 0.65:
        return f'<span class="risk-high">🔴  High Risk ({prob*100:.1f}%)</span>'
    elif prob >= 0.35:
        return f'<span class="risk-medium">🟡  Medium Risk ({prob*100:.1f}%)</span>'
    else:
        return f'<span class="risk-low">🟢  Low Risk ({prob*100:.1f}%)</span>'


# ── Sidebar — Customer Input ─────────────────────────────────────────────────
st.sidebar.header("🧑 Customer Details")
st.sidebar.markdown("Adjust the sliders to profile a customer.")

geography   = st.sidebar.selectbox("Geography",        ["France", "Germany", "Spain"])
gender      = st.sidebar.radio("Gender",               ["Male", "Female"])
age         = st.sidebar.slider("Age",                 18, 92, 38)
credit_score= st.sidebar.slider("Credit Score",        300, 850, 650)
tenure      = st.sidebar.slider("Tenure (years)",      0, 10, 5)
balance     = st.sidebar.number_input("Account Balance ($)", 0, 300000, 75000, step=5000)
num_products= st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cc      = st.sidebar.checkbox("Has Credit Card?",  value=True)
is_active   = st.sidebar.checkbox("Is Active Member?", value=True)
salary      = st.sidebar.number_input("Estimated Salary ($)", 10000, 250000, 100000, step=5000)
model_choice= st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))


# ── Main Panel ───────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📉 Bank Churn Forecaster</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict if a customer will leave the bank - CodSoft Task 3</p>', unsafe_allow_html=True)
st.divider()

col_pred, col_info = st.columns([1.2, 1])

with col_pred:
    st.subheader("Live Prediction")
    model = load_model(model_choice)

    if model is None:
        st.warning(
            f"⚠️  Model **{model_choice}** not found in `models/`. "
            "Run `python main.py` first to train and save all models."
        )
    else:
        inputs = dict(
            geography=geography, gender=gender, age=age,
            credit_score=credit_score, tenure=tenure, balance=balance,
            num_products=num_products, has_credit_card=int(has_cc),
            is_active_member=int(is_active), salary=salary,
        )
        X = build_feature_vector(inputs)
        prob = model.predict_proba(X)[0, 1]

        st.markdown(risk_label(prob), unsafe_allow_html=True)
        st.progress(float(prob))

        st.markdown(f"""
        <div class="metric-card">
            <b>Churn Probability:</b> {prob*100:.2f}%<br>
            <b>Model Used:</b> {model_choice}<br>
            <b>Decision:</b> {'Will likely churn' if prob >= 0.5 else 'Likely to stay'}
        </div>
        """, unsafe_allow_html=True)

        # Mini gauge chart
        fig, ax = plt.subplots(figsize=(4, 0.5))
        ax.barh(0, prob,       color="#E74C3C" if prob >= 0.5 else "#27AE60", height=0.5)
        ax.barh(0, 1 - prob, left=prob, color="#ECF0F1", height=0.5)
        ax.set_xlim(0, 1); ax.axis("off")
        ax.set_facecolor("#FAFAFA"); fig.set_facecolor("#FAFAFA")
        st.pyplot(fig, use_container_width=True)

with col_info:
    st.subheader("Customer Profile")
    profile_data = {
        "Field": ["Geography", "Gender", "Age", "Credit Score", "Tenure",
                  "Balance", "Products", "Credit Card", "Active Member", "Salary"],
        "Value": [geography, gender, age, credit_score, tenure,
                  f"${balance:,}", num_products,
                  "Yes" if has_cc else "No",
                  "Yes" if is_active else "No",
                  f"${salary:,}"],
    }
    st.dataframe(pd.DataFrame(profile_data).set_index("Field"), use_container_width=True)

st.divider()

# ── Batch Prediction via Upload ───────────────────────────────────────────────
st.subheader("📂 Batch Prediction (Upload CSV)")
uploaded = st.file_uploader("Upload a customer CSV (same format as Churn_Modelling.csv)", type=["csv"])

if uploaded is not None:
    df_batch = pd.read_csv(uploaded)
    st.write(f"Loaded {len(df_batch):,} rows.")
    st.dataframe(df_batch.head())

    model_batch = load_model(model_choice)
    if model_batch:
        try:
            from src.preprocess import preprocess
            # Quick hack: add a dummy target column if not present
            if "Exited" not in df_batch.columns:
                df_batch["Exited"] = 0

            _, X_b, _, _, feat, _ = preprocess(df_batch, test_size=0.9999, random_state=42)
            probs = model_batch.predict_proba(X_b)[:, 1]
            df_batch["ChurnProbability"] = np.nan
            df_batch.loc[X_b.index, "ChurnProbability"] = probs
            df_batch["Prediction"] = (probs >= 0.5).astype(int)

            st.success("Predictions complete!")
            st.dataframe(df_batch[["ChurnProbability", "Prediction"]].head(20))

            csv_out = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button("⬇  Download Predictions", csv_out, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("CodSoft ML Internship · Task 3 — Customer Churn Prediction · Built with scikit-learn & Streamlit")
