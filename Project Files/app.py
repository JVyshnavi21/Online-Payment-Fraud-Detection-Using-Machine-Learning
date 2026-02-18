import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Online Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model, le = pickle.load(f)
    return model, le

model, le = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Fraud Prediction", "Model Dashboard", "About Project"]
)

# =====================================================
# PAGE 1 — FRAUD PREDICTION
# =====================================================
if page == "Fraud Prediction":

    st.title("💳 Online Fraud Detection System")
    st.markdown("### Enter transaction details")

    col1, col2 = st.columns(2)

    with col1:
        type_input = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
        )
        amount = st.number_input("Amount", min_value=0.0)
        oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0)

    with col2:
        newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0)
        oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0)
        newbalanceDest = st.number_input("New Balance Destination", min_value=0.0)

    if st.button("🚀 Predict Fraud", use_container_width=True):

        type_encoded = le.transform([type_input])[0]

        input_df = pd.DataFrame([{
            'step': 1,
            'type': type_encoded,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest
        }])
        if 'step' in input_df.columns:
            input_df = input_df.drop(columns=['step'])
           
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # ---------- Gauge ----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Fraud Probability (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

        # ---------- SHAP ----------
        st.subheader("🧠 Model Explanation")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            fig2, ax = plt.subplots()
            shap.summary_plot(
                shap_values,
                input_df,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig2)
        except Exception:
            st.info("SHAP explanation not available in this environment.")

# =====================================================
# PAGE 2 — DASHBOARD
# =====================================================
elif page == "Model Dashboard":

    st.title("📊 Model Performance Dashboard")

    df = pd.read_csv("fraud.csv")
    fraud_count = df['isFraud'].value_counts()

    fig = go.Figure(data=[
        go.Pie(
            labels=["Legitimate", "Fraud"],
            values=fraud_count.values,
            hole=.4
        )
    ])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

# =====================================================
# PAGE 3 — ABOUT
# =====================================================
else:

    st.title("ℹ️ About This Project")

    st.markdown("""
### 💳 Online Fraud Detection

This project uses Machine Learning to detect fraudulent
financial transactions in real time.

### 🔧 Technologies Used
- Python  
- Scikit-learn  
- Streamlit  
- SHAP Explainability  
- Plotly  

### 🎯 Author
Vyshnavi Reddy

---
⭐ Internship-ready AI project
""")