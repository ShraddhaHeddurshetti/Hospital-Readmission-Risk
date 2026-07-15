import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1500px;
        margin: 0 auto;
    }
    .main-header {
        background: linear-gradient(90deg, #0E7C7B, #14A5A3);
        padding: 18px 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-header h1 { margin: 0; font-size: 26px; }
    .main-header p { margin: 4px 0 0 0; font-size: 14px; opacity: 0.9; }
    .risk-box-low {
        background-color: #E4F7E9;
        border-left: 6px solid #2E7D32;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .risk-box-high {
        background-color: #FDEBEC;
        border-left: 6px solid #C62828;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .risk-box-low h2, .risk-box-high h2 { margin: 0; font-size: 22px; }
    .risk-box-low p, .risk-box-high p { margin: 4px 0 0 0; font-size: 13px; }
    .section-card {
        background-color: white;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #D9F0EF;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏥 Hospital Readmission Risk Predictor</h1>
    
</div>
""", unsafe_allow_html=True)

model = joblib.load("notebook/readmission_model.pkl")
columns = joblib.load("notebook/model_columns.pkl")
defaults = joblib.load("notebook/default_values.pkl")

left_col, right_col = st.columns([1, 1.3], gap="medium")

with left_col:
    with st.container(border=True):
     st.subheader("📋 Patient Details")

    number_inpatient = st.slider("🛏️ Prior inpatient admissions", 0, 10, 0)
    number_emergency = st.slider("🚑 Prior emergency visits", 0, 10, 0)
    number_diagnoses = st.slider("📄 Number of diagnoses recorded", 1, 16, 5)
    time_in_hospital = st.slider("📅 Days spent in hospital (this visit)", 1, 14, 3)
    num_medications = st.slider("💊 Number of medications given", 1, 50, 10)
    diabetes_med = st.selectbox("🩺 On diabetes medication?", ["No", "Yes"])

    predict_clicked = st.button("🔍 Predict Risk", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

input_row = defaults.copy()
input_row["number_inpatient"] = number_inpatient
input_row["number_emergency"] = number_emergency
input_row["number_diagnoses"] = number_diagnoses
input_row["time_in_hospital"] = time_in_hospital
input_row["num_medications"] = num_medications
if "diabetesMed_Yes" in input_row.index:
    input_row["diabetesMed_Yes"] = 1 if diabetes_med == "Yes" else 0

X_input = pd.DataFrame([input_row])[columns]

with right_col:
    
    if predict_clicked:
        proba = model.predict_proba(X_input)[:, 1][0]

        if proba >= 0.5:
            st.markdown(f"""
            <div class="risk-box-high">
                <h2>⚠️ HIGH RISK: {proba*100:.1f}%</h2>
                <p>Elevated risk of 30-day readmission. Consider closer discharge planning.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-box-low">
                <h2>✅ LOW RISK: {proba*100:.1f}%</h2>
                <p>Lower risk of 30-day readmission based on current indicators.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**🧠 Why this prediction?**")
        st.caption("Factors pushing risk up (red) or down (blue) for this patient.")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        fig, ax = plt.subplots(figsize=(9, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_input.iloc[0],
                feature_names=columns
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("👈 Set patient details and click **Predict Risk** to see results here.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#888; font-size:12px; margin-top:15px;">
    Built with XGBoost + SHAP | Diabetes 130-US Hospitals Dataset
</div>
""", unsafe_allow_html=True)