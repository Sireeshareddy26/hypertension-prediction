import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load the saved model and preprocessors ---
model = joblib.load('model.joblib')
categorical_mappings = joblib.load('categorical_mappings.joblib')
feature_columns = joblib.load('feature_columns.joblib')

# Extract individual mappings for convenience
mapping_gender = categorical_mappings['Gender']
mapping_bmi_category = categorical_mappings['BMI Category']
mapping_aip_category = categorical_mappings['AIP Category']
mapping_tghdl_category = categorical_mappings['TG/HDL Category']
mapping_hypertension = categorical_mappings['Hypertension']

# Create inverse mapping for hypertension prediction display
inverse_mapping_hypertension = {v: k for k, v in mapping_hypertension.items()}

# --- 2. Define calculate_derived_lipid_indices function ---
def calculate_derived_lipid_indices(total_cholesterol_tc, ldl, hdl, triglycerides):
    # Ensure HDL is not zero to avoid division by zero in log and ratio calculations
    hdl_safe = hdl if hdl != 0 else 0.01 # Use a small positive value to prevent errors

    aip = np.log(triglycerides / hdl_safe) if hdl_safe != 0 and triglycerides > 0 else np.nan
    cr1_tc_per_hdl = total_cholesterol_tc / hdl_safe if hdl_safe != 0 else np.nan
    cr2_ldl_per_hdl = ldl / hdl_safe if hdl_safe != 0 else np.nan
    tg_hdl_ratio = triglycerides / hdl_safe if hdl_safe != 0 else np.nan
    ac_tc_hdl_per_hdl = (total_cholesterol_tc - hdl) / hdl_safe if hdl_safe != 0 else np.nan

    return {
        'AIP': aip,
        'CR1_TC_per_HDL': cr1_tc_per_hdl,
        'CR2_LDL_per_HDL': cr2_ldl_per_hdl,
        'TG/HDL Ratio': tg_hdl_ratio,
        'AC_TC-HDL_per_HDL': ac_tc_hdl_per_hdl
    }

# --- 3. Define inverse_map_hypertension function ---
def inverse_map_hypertension(prediction):
    return inverse_mapping_hypertension.get(prediction, "Unknown")

# --- 4. Streamlit UI ---
st.set_page_config(page_title='Hypertension Prediction App', layout='wide')
st.title('Hypertension Prediction Application')

st.markdown("""
    This application predicts the likelihood of hypertension based on various health indicators.
    Please input the patient's data below.
""")

# Input fields for features
input_data = {}

st.header("Demographics & Anthropometry")
col1, col2, col3 = st.columns(3)
with col1:
    input_data['Gender'] = st.selectbox('Gender', options=list(mapping_gender.keys()))
    input_data['Weight'] = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    input_data['BMI Category'] = st.selectbox('BMI Category', options=list(mapping_bmi_category.keys()))
with col2:
    input_data['Age'] = st.number_input('Age (years)', min_value=18, max_value=90, value=45, step=1)
    input_data['Height'] = st.number_input('Height (cm)', min_value=100.0, max_value=220.0, value=170.0, step=0.1)
with col3:
    input_data['BMI'] = st.number_input('BMI (kg/m²)', min_value=15.0, max_value=50.0, value=24.0, step=0.1)

st.header("Renal & Metabolic Markers")
col1, col2, col3 = st.columns(3)
with col1:
    input_data['Serum Creatinine'] = st.number_input('Serum Creatinine (mg/dL)', min_value=0.4, max_value=3.0, value=0.9, step=0.01)
    input_data['Serum Potassium'] = st.number_input('Serum Potassium (mEq/L)', min_value=3.0, max_value=6.0, value=4.0, step=0.01)
    input_data['Serum Albumin (g/dL)'] = st.number_input('Serum Albumin (g/dL)', min_value=2.0, max_value=5.5, value=4.0, step=0.1)
with col2:
    input_data['Serum Uric Acid'] = st.number_input('Serum Uric Acid (mg/dL)', min_value=2.0, max_value=10.0, value=5.0, step=0.1)
    input_data['Serum Sodium'] = st.number_input('Serum Sodium (mEq/L)', min_value=125.0, max_value=155.0, value=140.0, step=0.1)
    input_data['Albumin/Creatinine Ratio'] = st.number_input('Albumin/Creatinine Ratio', min_value=0.0, max_value=500.0, value=50.0, step=0.1)

st.header("Lipid Panel")
col1, col2, col3 = st.columns(3)
with col1:
    input_data['Total Cholesterol - TC'] = st.number_input('Total Cholesterol - TC (mg/dL)', min_value=100.0, max_value=400.0, value=180.0, step=1.0)
    input_data['HDL'] = st.number_input('HDL (mg/dL)', min_value=20.0, max_value=100.0, value=50.0, step=1.0)
    input_data['AIP Category'] = st.selectbox('AIP Category', options=list(mapping_aip_category.keys()))
with col2:
    input_data['LDL'] = st.number_input('LDL (mg/dL)', min_value=50.0, max_value=300.0, value=100.0, step=1.0)
    input_data['Triglycerides'] = st.number_input('Triglycerides (mg/dL)', min_value=50.0, max_value=1000.0, value=120.0, step=1.0)
    input_data['TG/HDL Category'] = st.selectbox('TG/HDL Category', options=list(mapping_tghdl_category.keys()))


if st.button('Predict Hypertension'):
    # Convert categorical string inputs to numerical encoded values
    processed_inputs = {}
    for k, v in input_data.items():
        if k in mapping_gender:
            processed_inputs[k] = mapping_gender[v]
        elif k in mapping_bmi_category:
            processed_inputs[k] = mapping_bmi_category[v]
        elif k in mapping_aip_category:
            processed_inputs[k] = mapping_aip_category[v]
        elif k in mapping_tghdl_category:
            processed_inputs[k] = mapping_tghdl_category[v]
        else:
            processed_inputs[k] = v

    # Calculate derived lipid indices
    derived_indices = calculate_derived_lipid_indices(
        processed_inputs['Total Cholesterol - TC'],
        processed_inputs['LDL'],
        processed_inputs['HDL'],
        processed_inputs['Triglycerides']
    )
    processed_inputs.update(derived_indices)

    # Combine all inputs into a pandas DataFrame, ensuring correct column order
    input_df = pd.DataFrame([processed_inputs], columns=feature_columns)

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # Display prediction and derived lipid indices
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"The model predicts: **{inverse_map_hypertension(prediction)}**")
        st.write(f"Probability of Hypertension: {prediction_proba[1]:.2f}")
        st.write(f"Probability of No Hypertension: {prediction_proba[0]:.2f}")
    else:
        st.success(f"The model predicts: **{inverse_map_hypertension(prediction)}**")
        st.write(f"Probability of No Hypertension: {prediction_proba[0]:.2f}")
        st.write(f"Probability of Hypertension: {prediction_proba[1]:.2f}")

    st.subheader("Calculated Derived Lipid Indices")
    st.write(f"**AIP (Atherogenic Index of Plasma):** {derived_indices['AIP']:.4f}")
    st.write(f"**CR1 (Total Cholesterol/HDL):** {derived_indices['CR1_TC_per_HDL']:.4f}")
    st.write(f"**CR2 (LDL/HDL):** {derived_indices['CR2_LDL_per_HDL']:.4f}")
    st.write(f"**TG/HDL Ratio:** {derived_indices['TG/HDL Ratio']:.4f}")
    st.write(f"**AC ((TC-HDL)/HDL):** {derived_indices['AC_TC-HDL_per_HDL']:.4f}")
