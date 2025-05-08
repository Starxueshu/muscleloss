# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import shap

st.header("An artificial intelligence tool to predict postoperative muscle loss among advanced cancer patients treated with surgery for metastatic spinal disease: an establishment and validation study")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

diabetes = st.sidebar.selectbox("Diabetes", ("No", "Yes"))
chronic_pulmonary_disease = st.sidebar.selectbox("Chronic pulmonary disease", ("No", "Yes"))
ECOG = st.sidebar.selectbox("ECOG score", ("One", "Two", "Three", "Four"))
custom_surgerytpe = st.sidebar.selectbox("Surgery type", ("Minimal invasive", "Open"))
custom_corticosteroids = st.sidebar.selectbox("Corticosteroids", ("No", "Yes"))
neutrophils_abs_first = st.sidebar.slider("Neutrophils (×10^9/L)", 0.00, 15.00)
albumin_first = st.sidebar.slider("Albumin (g/dL)", 1.00, 8.00)
creatinine_first = st.sidebar.slider("Creatinine (mg/dL)", 0.00, 6.00)
bun_first = st.sidebar.slider("BUN (mg/dL)", 10.00, 40.00)
calcium_total_first = st.sidebar.slider("Total calcium (mg/dL)", 2.00, 16.00)

if st.button("Submit"):
    rf_clf = jl.load("lightgbm_clf_final_roundweb.pkl")
    x = pd.DataFrame([[diabetes, chronic_pulmonary_disease, ECOG, custom_surgerytpe, custom_corticosteroids, neutrophils_abs_first, albumin_first, creatinine_first, bun_first, calcium_total_first]],
                     columns=["diabetes", "chronic_pulmonary_disease", "ECOG", "custom_surgerytpe", "custom_corticosteroids", "neutrophils_abs_first", "albumin_first", "creatinine_first", "bun_first", "calcium_total_first"])
    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["One", "Two", "Three", "Four"], [1, 2, 3, 4])
    x = x.replace(["Minimal invasive", "Open"], [0, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Postoperative muscle loss: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.50:
        st.success(f"Risk group: low-risk group")
    else:
        st.error(f"Risk group: High-risk group")

st.subheader('About the model')
st.markdown('This study presents a clinically artificial intelligence (AI) application designed to predict postoperative muscle loss in patients undergoing surgery for metastatic spinal disease. Leveraging the Light Gradient Boosting Machine (LightGBM) algorithm—which demonstrated superior performance (AUC: 0.820, accuracy: 78.3%) among six evaluated machine learning models. External validation confirmed robust generalizability (AUC: 0.805).')