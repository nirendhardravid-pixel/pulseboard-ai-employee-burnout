import streamlit as st
import pandas as pd
import joblib

# ============================================================
# Load the trained model (must be in the same folder as this file)
# ============================================================
model = joblib.load('burnout_model.pkl')

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="PulseBoard AI", page_icon="🔥", layout="centered")
st.title("🔥 PulseBoard AI")
st.subheader("Employee Burnout Risk Predictor")
st.write("Enter employee details below to predict their burnout risk.")

st.divider()

# ============================================================
# TAB 1: Single Employee Prediction
# ============================================================
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Bulk Prediction (CSV)"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        designation = st.slider("Designation Level (0=Junior, 5=Executive)", 0.0, 5.0, 2.0, 0.5)
        resource_alloc = st.slider("Resource Allocation (Workload)", 1.0, 10.0, 5.0, 0.5)
        mental_fatigue = st.slider("Mental Fatigue Score", 0.0, 10.0, 5.0, 0.5)
        tenure_days = st.number_input("Tenure (Days at Company)", min_value=0, value=365)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        company_type = st.selectbox("Company Type", ["Service", "Product"])
        wfh = st.selectbox("WFH Setup Available", ["Yes", "No"])

    predict_btn = st.button("🔮 Predict Burnout Risk", use_container_width=True)

    if predict_btn:
        # Build the input row in EXACTLY the same column order/format as X_train
        input_df = pd.DataFrame({
            'Designation': [designation],
            'Resource Allocation': [resource_alloc],
            'Mental Fatigue Score': [mental_fatigue],
            'Tenure_Days': [tenure_days],
            'Gender_Female': [1 if gender == 'Female' else 0],
            'Gender_Male': [1 if gender == 'Male' else 0],
            'Company Type_Product': [1 if company_type == 'Product' else 0],
            'Company Type_Service': [1 if company_type == 'Service' else 0],
            'WFH Setup Available_No': [1 if wfh == 'No' else 0],
            'WFH Setup Available_Yes': [1 if wfh == 'Yes' else 0],
        })

        pred = model.predict(input_df)[0]
        pred = max(0, min(1, pred))  # keep within 0-1 range for display

        if pred < 0.35:
            risk, color = "Low", "green"
        elif pred < 0.65:
            risk, color = "Medium", "orange"
        else:
            risk, color = "High", "red"

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Predicted Burn Rate", f"{pred:.2f}")
        c2.markdown(f"### Risk Level: :{color}[{risk}]")
        st.progress(pred)

# ============================================================
# TAB 2: Bulk Prediction from CSV
# ============================================================
with tab2:
    st.write("Upload a CSV with the same columns as the HackerEarth Burnout dataset "
             "(Employee ID, Date of Joining, Gender, Company Type, WFH Setup Available, "
             "Designation, Resource Allocation, Mental Fatigue Score).")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        df_original = df_raw.copy()

        # ---- Same cleaning pipeline used in training ----
        df_raw = df_raw.drop_duplicates()

        df_raw['Date of Joining'] = pd.to_datetime(df_raw['Date of Joining'])
        df_raw['Tenure_Days'] = (pd.Timestamp('2024-01-01') - df_raw['Date of Joining']).dt.days
        df_raw = df_raw.drop(['Employee ID', 'Date of Joining'], axis=1)

        df_raw = pd.get_dummies(df_raw, columns=['Gender', 'Company Type', 'WFH Setup Available'])

        # Align columns to exactly what the model expects
        expected_cols = ['Designation', 'Resource Allocation', 'Mental Fatigue Score', 'Tenure_Days',
                          'Gender_Female', 'Gender_Male', 'Company Type_Product', 'Company Type_Service',
                          'WFH Setup Available_No', 'WFH Setup Available_Yes']
        for c in expected_cols:
            if c not in df_raw.columns:
                df_raw[c] = 0
        df_raw = df_raw[expected_cols]

        predictions = model.predict(df_raw)

        result_df = df_original.loc[df_raw.index].copy()
        result_df['Predicted_Burn_Rate'] = predictions
        result_df['Burnout_Risk'] = pd.cut(
            result_df['Predicted_Burn_Rate'],
            bins=[0, 0.35, 0.65, 1.0],
            labels=['Low', 'Medium', 'High']
        )

        st.success(f"Predictions generated for {len(result_df)} employees!")
        st.dataframe(result_df.head(20))

        st.bar_chart(result_df['Burnout_Risk'].value_counts())

        csv_output = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Full Results as CSV",
            data=csv_output,
            file_name="burnout_predictions.csv",
            mime="text/csv"
        )

st.divider()
st.caption("PulseBoard AI — Built with Random Forest + SHAP explainability | Portfolio Project")