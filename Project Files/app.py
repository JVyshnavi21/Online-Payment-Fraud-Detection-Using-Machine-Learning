import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
)

# =============================
# THEME TOGGLE
# =============================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = (
        "light" if st.session_state.theme == "dark" else "dark"
    )

# =============================
# FAANG CSS
# =============================
if st.session_state.theme == "dark":
    bg = "#0b1220"
    card = "rgba(255,255,255,0.06)"
    text = "white"
else:
    bg = "#f5f7fb"
    card = "white"
    text = "#111"

st.markdown(
    f"""
<style>

/* App background */
.stApp {{
    background: {bg};
    color: {text};
}}

/* Top navbar */
.navbar {{
    position: sticky;
    top: 0;
    z-index: 999;
    padding: 14px 24px;
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    border-radius: 14px;
    margin-bottom: 18px;
}}

/* Title */
.nav-title {{
    font-size: 28px;
    font-weight: 800;
    color: white;
}}

/* Glass card */
.card {{
    background: {card};
    padding: 22px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    transition: 0.25s;
}}
.card:hover {{
    transform: translateY(-4px);
}}

/* KPI number */
.kpi {{
    font-size: 32px;
    font-weight: 800;
}}

/* Risk pills */
.pill-high {{
    background:#dc2626;
    padding:6px 16px;
    border-radius:999px;
    font-weight:700;
    color:white;
}}
.pill-med {{
    background:#f59e0b;
    padding:6px 16px;
    border-radius:999px;
    font-weight:700;
    color:white;
}}
.pill-low {{
    background:#16a34a;
    padding:6px 16px;
    border-radius:999px;
    font-weight:700;
    color:white;
}}

</style>
""",
    unsafe_allow_html=True,
)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model, le, features = pickle.load(f)
    return model, le, features

model, le, features = load_artifacts()

# =============================
# NAVBAR
# =============================
nav1, nav2 = st.columns([6, 1])

with nav1:
    st.markdown(
        '<div class="navbar"><span class="nav-title">🛡️ FraudShield AI</span></div>',
        unsafe_allow_html=True,
    )

with nav2:
    st.button("🌙 Toggle Theme", on_click=toggle_theme)

st.caption("Enterprise-grade payment risk intelligence")

# =============================
# SIDEBAR
# =============================
st.sidebar.header("🧾 Controls")

mode = st.sidebar.radio(
    "Mode",
    ["🔍 Single Analysis", "📂 Bulk Analytics"],
)

# =========================================================
# 🔹 SINGLE MODE
# =========================================================
if mode == "🔍 Single Analysis":

    step = st.sidebar.number_input("Step", 1, 1000, 1)
    type_input = st.sidebar.selectbox("Transaction Type", le.classes_)
    amount = st.sidebar.number_input("Amount", 0.0, value=1000.0)
    oldbalanceOrg = st.sidebar.number_input("Sender Old Balance", 0.0, value=5000.0)
    newbalanceOrig = st.sidebar.number_input("Sender New Balance", 0.0, value=4000.0)
    oldbalanceDest = st.sidebar.number_input("Receiver Old Balance", 0.0, value=0.0)
    newbalanceDest = st.sidebar.number_input("Receiver New Balance", 0.0, value=1000.0)

    run = st.sidebar.button("🚀 Run AI Analysis")

    # KPI ROW
    k1, k2, k3 = st.columns(3)

    k1.markdown(
        f'<div class="card"><div>💰 Amount</div><div class="kpi">₹{amount:,.0f}</div></div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div class="card"><div>📤 Sender</div><div class="kpi">₹{oldbalanceOrg:,.0f}</div></div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div class="card"><div>📥 Receiver</div><div class="kpi">₹{newbalanceDest:,.0f}</div></div>',
        unsafe_allow_html=True,
    )

    # =============================
    # PREDICTION
    # =============================
    if run:

        type_encoded = le.transform([type_input])[0]

        input_dict = {
            "step": step,
            "type": type_encoded,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
        }

        input_df = pd.DataFrame([input_dict])[features]

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        # Risk pill
        if prob > 0.8:
            pill = '<span class="pill-high">HIGH RISK</span>'
        elif prob > 0.4:
            pill = '<span class="pill-med">MEDIUM RISK</span>'
        else:
            pill = '<span class="pill-low">LOW RISK</span>'

        st.markdown("## 🤖 AI Verdict")

        v1, v2 = st.columns([1, 1])

        with v1:
            if pred == 1:
                st.error("🚨 Fraudulent Transaction")
            else:
                st.success("✅ Legitimate Transaction")

            st.markdown(f"### Risk Level: {pill}", unsafe_allow_html=True)
            st.progress(float(prob))

        # Donut gauge
        with v2:
            fig = go.Figure(
                go.Pie(
                    values=[prob, 1 - prob],
                    hole=0.7,
                    labels=["Fraud Risk", ""],
                    marker=dict(colors=["#ef4444", "#1f2937"]),
                    textinfo="none",
                )
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 🔹 BULK MODE
# =========================================================
else:

    st.subheader("📂 Bulk Intelligence")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "type" in df.columns:
            df["type"] = le.transform(df["type"])

        input_df = df[features]

        df["Prediction"] = model.predict(input_df)
        df["Fraud_Probability"] = model.predict_proba(input_df)[:, 1]

        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(
                px.pie(df, names="Prediction", title="Fraud Distribution"),
                use_container_width=True,
            )

        with c2:
            st.plotly_chart(
                px.scatter(
                    df,
                    x="amount",
                    y="Fraud_Probability",
                    color="Prediction",
                    title="Risk Scatter",
                ),
                use_container_width=True,
            )

        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv, "fraud_predictions.csv")

st.markdown("---")
st.caption("🛡️ FraudShield AI • FAANG-level UI • Built by Vyshnavi")
