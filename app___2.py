import streamlit as st
from cancer_predictor import predict_risk
import re

st.title("Cancer Prediction App")
st.subheader("Ask a question to assess cancer risk")

user_input = st.text_input("Describe the patient (e.g., '60-year-old female who smokes and has chest pain')")

if st.button("Predict"):
    # Default input for all features the model expects
    input_dict = {
        "Age": 20,
        "Gender": "Male",
        "Air.Pollution": 1,
        "Alcohol.use": 1,
        "Dust.Allergy": 1,
        "OccuPational.Hazards": 1,
        "Genetic.Risk": 1,
        "chronic.Lung.Disease": 1,
        "Balanced.Diet": 1,
        "Obesity": 1,
        "Smoking": 1,
        "Passive.Smoker": 1,
        "Chest.Pain": 1,
        "Coughing.of.Blood": 1,
        "Fatigue": 1,
        "Weight.Loss": 1,
        "Shortness.of.Breath": 1,
        "Wheezing": 1,
        "Swallowing.Difficulty": 1,
        "Clubbing.of.Finger.Nails": 1,
        "Frequent.Cold": 1,
        "Dry.Cough": 1,
        "Snoring": 1
    }

    q = user_input.lower()

    # Gender
    if "female" in q:
        input_dict["Gender"] = "Female"
    elif "male" in q:
        input_dict["Gender"] = "Male"

    # Age
    age_match = re.search(r"(\d+)[ -]?(year|yr)[s]?[ -]?old", q)
    if age_match:
        input_dict["Age"] = int(age_match.group(1))

    # Smoking
    if "non-smoker" in q or "does not smoke" in q:
        input_dict["Smoking"] = 0
    elif "smokes" in q or "smoking" in q:
        input_dict["Smoking"] = 1

    # Passive Smoker
    if "passive smoker" in q or "exposed to smoke" in q:
        input_dict["Passive.Smoker"] = 1

    # Alcohol
    if "no alcohol" in q or "does not drink" in q:
        input_dict["Alcohol.use"] = 1
    elif "drinks alcohol" in q or "alcohol use" in q:
        input_dict["Alcohol.use"] = 3

    # Air Pollution
    if "pollution" in q or "air pollution" in q:
        input_dict["Air.Pollution"] = 4

    # Dust Allergy
    if "dust allergy" in q or "allergic to dust" in q:
        input_dict["Dust.Allergy"] = 3

    # Occupational Hazards
    if "chemical exposure" in q or "occupational hazard" in q:
        input_dict["OccuPational.Hazards"] = 3

    # Genetic Risk
    if "family history" in q or "genetic risk" in q:
        input_dict["Genetic.Risk"] = 3

    # Chronic Lung Disease
    if "chronic lung" in q or "lung disease" in q:
        input_dict["chronic.Lung.Disease"] = 3

    # Balanced Diet
    if "poor diet" in q or "unhealthy diet" in q:
        input_dict["Balanced.Diet"] = 0
    elif "healthy diet" in q or "balanced diet" in q:
        input_dict["Balanced.Diet"] =_


    # Age parsing
    age_match = re.search(r"(\d+)[ -]?(year|yr)[s]?[ -]?old", q)
    if age_match:
        input_dict["Age"] = int(age_match.group(1))

    # Predict using model
    try:
        result = predict_risk(input_dict)
        st.subheader("Prediction")
        st.markdown(f"**Risk Level:** {result['risk_level']}")
        st.markdown(f"**Insight:** {result['insight']}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    st.subheader("Input Sent to Model")
    st.json(input_dict)
