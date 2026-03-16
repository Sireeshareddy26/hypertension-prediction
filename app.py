import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the trained machine learning model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the feature columns and their order used during training
# This list was obtained from X.columns.tolist() in a previous step
feature_columns = ['Age (years)', 'Weight (kg)', 'Height (cm)', 'BMI (kg/m²)', 'Serum Creatinine (mg/dL)',
                   'Serum Uric Acid (mg/dL)', 'Serum Potassium (mEq/L)', 'Serum Sodium (mEq/L)',
                   'Serum Albumin (g/dL)', 'Albumin/Creatinine Ratio', 'Total Cholesterol - TC (mg/dL)',
                   'LDL (mg/dL)', 'HDL (mg/dL)', 'Triglycerides - TG (mg/dL)', 'AIP [log(TG/HDL)]',
                   'CR1 (TC/HDL)', 'CR2 (LDL/HDL)', 'TG/HDL Ratio', 'AC [(TC-HDL)/HDL]', 'Gender_Male',
                   'BMI Category_Obese', 'BMI Category_Overweight', 'BMI Category_Underweight',
                   'AIP Category_Intermediate Risk', 'AIP Category_Low Risk', 'TG/HDL Category_Ideal',
                   'TG/HDL Category_Moderate Risk']

# 2. Set the title of the Streamlit application
st.title('Hypertension Prediction Application')
st.write('Enter patient details to predict the likelihood of hypertension.')

# 3. Create Streamlit input widgets for each feature
with st.sidebar:
    st.header('Patient Input Features')

    # Numerical inputs
    age = st.number_input('Age (years)', min_value=1, max_value=100, value=45)
    weight = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, value=70.0, format="%.1f")
    height = st.number_input('Height (cm)', min_value=100.0, max_value=250.0, value=170.0, format="%.1f")
    bmi = st.number_input('BMI (kg/m²)', min_value=10.0, max_value=60.0, value=24.0, format="%.2f")
    serum_creatinine = st.number_input('Serum Creatinine (mg/dL)', min_value=0.1, max_value=10.0, value=0.9, format="%.2f")
    serum_uric_acid = st.number_input('Serum Uric Acid (mg/dL)', min_value=1.0, max_value=15.0, value=5.0, format="%.2f")
    serum_potassium = st.number_input('Serum Potassium (mEq/L)', min_value=2.0, max_value=7.0, value=4.0, format="%.2f")
    serum_sodium = st.number_input('Serum Sodium (mEq/L)', min_value=120.0, max_value=160.0, value=140.0, format="%.1f")
    serum_albumin = st.number_input('Serum Albumin (g/dL)', min_value=2.0, max_value=6.0, value=4.0, format="%.2f")
    albumin_creatinine_ratio = st.number_input('Albumin/Creatinine Ratio', min_value=0.1, max_value=10.0, value=4.0, format="%.2f")
    total_cholesterol = st.number_input('Total Cholesterol - TC (mg/dL)', min_value=100.0, max_value=400.0, value=200.0, format="%.1f")
    ldl = st.number_input('LDL (mg/dL)', min_value=30.0, max_value=300.0, value=120.0, format="%.1f")
    hdl = st.number_input('HDL (mg/dL)', min_value=20.0, max_value=100.0, value=50.0, format="%.1f")
    triglycerides = st.number_input('Triglycerides - TG (mg/dL)', min_value=50.0, max_value=500.0, value=150.0, format="%.1f")
    aip_log_tg_hdl = st.number_input('AIP [log(TG/HDL)]', min_value=-0.5, max_value=1.0, value=0.3, format="%.4f")
    cr1_tc_hdl = st.number_input('CR1 (TC/HDL)', min_value=1.0, max_value=10.0, value=4.0, format="%.4f")
    cr2_ldl_hdl = st.number_input('CR2 (LDL/HDL)', min_value=0.5, max_value=5.0, value=2.0, format="%.4f")
    tg_hdl_ratio = st.number_input('TG/HDL Ratio', min_value=0.5, max_value=10.0, value=3.0, format="%.4f")
    ac_tc_hdl = st.number_input('AC [(TC-HDL)/HDL]', min_value=0.5, max_value=10.0, value=3.0, format="%.4f")

    # Categorical inputs
    gender = st.selectbox('Gender', ['Female', 'Male'])
    bmi_category = st.selectbox('BMI Category', ['Normal', 'Overweight', 'Obese', 'Underweight'])
    aip_category = st.selectbox('AIP Category', ['High Risk', 'Intermediate Risk', 'Low Risk'])
    tg_hdl_category = st.selectbox('TG/HDL Category', ['Moderate Risk', 'High Risk', 'Ideal'])

# 4. Preprocess user inputs into a DataFrame matching model's training format
def preprocess_input(input_data):
    # Initialize a dictionary to store preprocessed data
    processed_data_dict = {
        'Age (years)': input_data['Age (years)'],
        'Weight (kg)': input_data['Weight (kg)'],
        'Height (cm)': input_data['Height (cm)'],
        'BMI (kg/m²)': input_data['BMI (kg/m²)'],
        'Serum Creatinine (mg/dL)': input_data['Serum Creatinine (mg/dL)'],
        'Serum Uric Acid (mg/dL)': input_data['Serum Uric Acid (mg/dL)'],
        'Serum Potassium (mEq/L)': input_data['Serum Potassium (mEq/L)'],
        'Serum Sodium (mEq/L)': input_data['Serum Sodium (mEq/L)'],
        'Serum Albumin (g/dL)': input_data['Serum Albumin (g/dL)'],
        'Albumin/Creatinine Ratio': input_data['Albumin/Creatinine Ratio'],
        'Total Cholesterol - TC (mg/dL)': input_data['Total Cholesterol - TC (mg/dL)'],
        'LDL (mg/dL)': input_data['LDL (mg/dL)'],
        'HDL (mg/dL)': input_data['HDL (mg/dL)'],
        'Triglycerides - TG (mg/dL)': input_data['Triglycerides - TG (mg/dL)'],
        'AIP [log(TG/HDL)]': input_data['AIP [log(TG/HDL)]'],
        'CR1 (TC/HDL)': input_data['CR1 (TC/HDL)'],
        'CR2 (LDL/HDL)': input_data['CR2 (LDL/HDL)'],
        'TG/HDL Ratio': input_data['TG/HDL Ratio'],
        'AC [(TC-HDL)/HDL]': input_data['AC [(TC-HDL)/HDL]'],
        'Gender_Male': False, # Default to False (Female)
        'BMI Category_Obese': False,
        'BMI Category_Overweight': False,
        'BMI Category_Underweight': False,
        'AIP Category_Intermediate Risk': False,
        'AIP Category_Low Risk': False,
        'TG/HDL Category_Ideal': False,
        'TG/HDL Category_Moderate Risk': False
    }

    # Populate one-hot encoded categorical features with boolean values
    if input_data['Gender'] == 'Male':
        processed_data_dict['Gender_Male'] = True

    if input_data['BMI Category'] == 'Obese':
        processed_data_dict['BMI Category_Obese'] = True
    elif input_data['BMI Category'] == 'Overweight':
        processed_data_dict['BMI Category_Overweight'] = True
    elif input_data['BMI Category'] == 'Underweight':
        processed_data_dict['BMI Category_Underweight'] = True

    if input_data['AIP Category'] == 'Intermediate Risk':
        processed_data_dict['AIP Category_Intermediate Risk'] = True
    elif input_data['AIP Category'] == 'Low Risk':
        processed_data_dict['AIP Category_Low Risk'] = True

    if input_data['TG/HDL Category'] == 'High Risk':
        # Note: 'High Risk' is the default for TG/HDL when drop_first=True and others are False
        # But if we explicitly set others, this would implicitly be handled.
        # For safety and clarity, if 'High Risk' is selected, we need to ensure the other two are False
        pass # 'High Risk' is the baseline when drop_first=True for TG/HDL Category
    elif input_data['TG/HDL Category'] == 'Ideal':
        processed_data_dict['TG/HDL Category_Ideal'] = True
    elif input_data['TG/HDL Category'] == 'Moderate Risk':
        processed_data_dict['TG/HDL Category_Moderate Risk'] = True

    # Create a DataFrame from the dictionary, ensuring column order
    processed_df = pd.DataFrame([processed_data_dict], columns=feature_columns)
    return processed_df

# Collect inputs into a dictionary
input_data = {
    'Age (years)': age,
    'Weight (kg)': weight,
    'Height (cm)': height,
    'BMI (kg/m²)': bmi,
    'Serum Creatinine (mg/dL)': serum_creatinine,
    'Serum Uric Acid (mg/dL)': serum_uric_acid,
    'Serum Potassium (mEq/L)': serum_potassium,
    'Serum Sodium (mEq/L)': serum_sodium,
    'Serum Albumin (g/dL)': serum_albumin,
    'Albumin/Creatinine Ratio': albumin_creatinine_ratio,
    'Total Cholesterol - TC (mg/dL)': total_cholesterol,
    'LDL (mg/dL)': ldl,
    'HDL (mg/dL)': hdl,
    'Triglycerides - TG (mg/dL)': triglycerides,
    'AIP [log(TG/HDL)]': aip_log_tg_hdl,
    'CR1 (TC/HDL)': cr1_tc_hdl,
    'CR2 (LDL/HDL)': cr2_ldl_hdl,
    'TG/HDL Ratio': tg_hdl_ratio,
    'AC [(TC-HDL)/HDL]': ac_tc_hdl,
    'Gender': gender,
    'BMI Category': bmi_category,
    'AIP Category': aip_category,
    'TG/HDL Category': tg_hdl_category
}

# 5. Add a 'Predict' button
if st.button('Predict Hypertension'):
    # Preprocess the input data
    processed_input = preprocess_input(input_data)

    # Make prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1]

    # 6. Display the prediction result
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'The patient is predicted to have Hypertension with a probability of {prediction_proba[0]:.2f}.')
    else:
        st.success(f'The patient is predicted to have No Hypertension with a probability of {1-prediction_proba[0]:.2f}.')

    st.write('---')
    st.write('### Input Data Summary:')
    st.write(pd.DataFrame([input_data]))
