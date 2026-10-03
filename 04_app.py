import streamlit as st
import pandas as pd
import joblib

# load the saved model and its column order
model = joblib.load("depression_tree_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title("Student Depression Risk Screener")
st.warning("This is a screening aid built for a college project. "
           "It is NOT a medical diagnosis.")

st.write("Answer three questions:")

thoughts = st.radio("Have you ever had suicidal thoughts?", ["No", "Yes"])
academic = st.slider("Academic pressure (1 = low, 5 = high)", 1, 5, 3)
financial = st.slider("Financial stress (1 = low, 5 = high)", 1, 5, 3)

if st.button("Check risk"):
    # build one row with all 22 columns; the tree only uses these three
    row = pd.DataFrame([[0] * len(columns)], columns=columns)
    row["Suicidal Thoughts"] = 1 if thoughts == "Yes" else 0
    row["Academic Pressure"] = academic
    row["Financial Stress"] = financial

    prediction = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]

    if prediction == 1:
        st.error(f"Higher risk (model score: {prob:.0%})")
        st.write("Please consider talking to a counselor, doctor, or someone "
                 "you trust. In India you can contact Tele-MANAS at 14416 "
                 "(please verify the number on the official site).")
    else:
        st.success(f"Lower risk (model score: {prob:.0%})")
        st.write("If you are struggling, it's still okay to reach out "
                 "to a counselor or someone you trust.")
