import streamlit as st
import joblib
import numpy as np

# Load trained pipeline
model = joblib.load('C:/Users/gihoz/OneDrive/Desktop/ReAdmitAi/dashboard/readmission_model.pkl')

st.title("🏥 Patient Readmission Predictor (<30 days)")

with st.form("readmit_form"):
    age = st.number_input("Age", min_value=0, max_value=10)
    time_in_hospital = st.number_input("Time in hospital", min_value=0)
    num_lab_procedures = st.number_input("Number of lab procedures", min_value=0)
    num_procedures = st.number_input("Number of procedures", min_value=0)
    num_medications = st.number_input("Number of medications", min_value=0)
    number_outpatient = st.number_input("Number of outpatient visits", min_value=0)
    number_emergency = st.number_input("Number of emergency visits", min_value=0)
    number_inpatient = st.number_input("Number of inpatient visits", min_value=0)
    number_diagnoses = st.number_input("Number of diagnoses", min_value=0)

    race = st.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'])
    gender = st.selectbox("Gender", ['Male', 'Female', 'Unknown/Invalid'])
    admission_type_id = st.selectbox("Admission Type ID", ['1', '2', '3', '4', '5', '6', '7', '8'])

    insulin = st.selectbox("Insulin", ['No', 'Yes', 'Up', 'Down'])
    change = st.selectbox("Change in meds", ['No', 'Ch'])
    diabetesMed = st.selectbox("Diabetes medication", ['Yes', 'No'])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = [
        age, time_in_hospital, num_lab_procedures, num_procedures,
        num_medications, number_outpatient, number_emergency,
        number_inpatient, number_diagnoses,
        race, gender, admission_type_id,
        insulin, change, diabetesMed
    ]

    import pandas as pd

    feature_columns = [
        'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses',
        'race', 'gender', 'admission_type_id',
        'insulin', 'change', 'diabetesMed'
    ]

    input_df = pd.DataFrame([input_data], columns=feature_columns)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ The patient is likely to be readmitted within 30 days.")
    else:
        st.info("🟦 The patient is NOT likely to be readmitted within 30 days.")


    