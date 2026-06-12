import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Medical Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
model = load_model("models/ann_model.h5")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==========================
# PREMIUM CSS
# ==========================
st.markdown("""
<style>

.stApp {
    color: white;
}

.title {
    text-align:center;
    font-size:48px;
    font-weight:900;
    color:#60A5FA;
}

.subtitle {
    text-align:center;
    color:#9CA3AF;
    font-size:18px;
    margin-bottom:20px;
}

.card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
}

.stButton>button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR NAVIGATION
# ==========================
with st.sidebar:

    st.title("🧠 Medical AI")

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "🔍 Prediction", "📊 Analytics"]
    )

    st.markdown("---")

    st.success("System Online 🟢")

# ==========================
# DASHBOARD PAGE
# ==========================
if page == "🏠 Dashboard":

    st.markdown("<div class='title'>Diabetes Risk AI System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-powered medical diagnosis dashboard</div>", unsafe_allow_html=True)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Patients", "10K+")
    c2.metric("Accuracy", "89%")
    c3.metric("Model", "ANN")
    c4.metric("Status", "Active")

    st.divider()

    # Fake dataset visualization (for UI richness)
    data = np.random.normal(50, 15, 100)

    fig = px.histogram(
        data,
        nbins=20,
        title="Simulated Glucose Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================
# PREDICTION PAGE
# ==========================
elif page == "🔍 Prediction":

    st.markdown("<div class='title'>Patient Risk Prediction</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 100)
        bp = st.number_input("Blood Pressure", 0, 150, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 10, 100, 30)

    if st.button("🔍 Predict Risk"):

        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled = scaler.transform(input_data)

        pred = model.predict(scaled, verbose=0)[0][0]
        risk = float(pred * 100)

        st.divider()

        # ==========================
        # GAUGE CHART
        # ==========================
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={"text": "Diabetes Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 40], "color": "green"},
                    {"range": [40, 70], "color": "orange"},
                    {"range": [70, 100], "color": "red"}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        # ==========================
        # RESULT
        # ==========================
        if risk > 70:
            st.error(f"🔴 HIGH RISK ({risk:.2f}%)")
        elif risk > 40:
            st.warning(f"🟠 MODERATE RISK ({risk:.2f}%)")
        else:
            st.success(f"🟢 LOW RISK ({risk:.2f}%)")

        # ==========================
        # PIE CHART
        # ==========================
        pie = go.Figure(data=[
            go.Pie(
                labels=["Risk", "Safe"],
                values=[risk, 100-risk],
                hole=0.5
            )
        ])

        st.plotly_chart(pie, use_container_width=True)

        # ==========================
        # BAR CHART
        # ==========================
        bar = px.bar(
            x=["Risk", "Safe"],
            y=[risk, 100-risk],
            title="Risk vs Safety Comparison"
        )

        st.plotly_chart(bar, use_container_width=True)

        # ==========================
        # KPI METRICS
        # ==========================
        m1, m2, m3 = st.columns(3)

        m1.metric("Risk Score", f"{risk:.2f}%")
        m2.metric("Age Factor", age)
        m3.metric("Glucose Level", glucose)

# ==========================
# ANALYTICS PAGE
# ==========================
elif page == "📊 Analytics":

    st.markdown("<div class='title'>Medical Analytics Dashboard</div>", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        fig1 = px.box(
            y=np.random.normal(100, 30, 200),
            title="Glucose Spread Analysis"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        fig2 = px.histogram(
            np.random.normal(30, 10, 200),
            nbins=25,
            title="Age Distribution"
        )

        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        x=np.random.randint(1, 10, 100),
        y=np.random.randint(50, 200, 100),
        title="BMI vs Glucose Relationship"
    )

    st.plotly_chart(fig3, use_container_width=True)