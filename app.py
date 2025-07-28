import streamlit as st
from cancer_predictor import predict_risk
import re

st.title("Cancer Prediction App")
st.subheader("Ask a question to assess cancer risk")

st.text("Features include: Age, Gender, Air Pollution, Alcohol use, Dust Allergy, OccuPational Hazards, Genetic Risk, Chronic Lung Disease, Balanced Diet, Obesity, Smoking, Passive Smoker, Chest Pain, Coughing of Blood, Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring")

user_input = st.text_input("Describe the patient (e.g., '60-year-old female, smoking heavily and chest pain is moderate')")

if st.button("Predict"):
    # Default input
    input_dict = {
        "Age": 20,
        "Gender": "Male",
        "Air Pollution": 1,
        "Alcohol use": 1,
        "Dust Allergy": 1,
        "OccuPational Hazards": 1,
        "Genetic Risk": 1,
        "Chronic Lung Disease": 1,
        "Balanced Diet": 1,
        "Obesity": 1,
        "Smoking": 1,
        "Passive Smoker": 1,
        "Chest Pain": 1,
        "Coughing of Blood": 1,
        "Fatigue": 1,
        "Weight Loss": 1,
        "Shortness of Breath": 1,
        "Wheezing": 1,
        "Swallowing Difficulty": 1,
        "Clubbing of Finger Nails": 1,
        "Frequent Cold": 1,
        "Dry Cough": 1,
        "Snoring": 1
    }

    q = user_input.lower()

    # Gender
    if "female" in q:
        input_dict["Gender"] = "Female"
    elif "male" in q:
        input_dict["Gender"] = "Male"

    risk_map = {
        "very low": 1,
        "low": 2,
        "moderate": 5,
        "medium": 5,
        "high": 8,
        "heavily": 8,
        "very high": 9
    }

    fields = [
        "Air Pollution", "Alcohol use", "Dust Allergy", "OccuPational Hazards",
        "Genetic Risk", "Chronic Lung Disease", "Balanced Diet", "Obesity", 
        "Smoking", "Passive Smoker", "Chest Pain", "Coughing of Blood", "Fatigue", 
        "Weight Loss", "Shortness of Breath", "Wheezing", "Swallowing Difficulty", 
        "Clubbing of Finger Nails", "Frequent Cold", "Dry Cough", "Snoring"
    ]

    # Set all features to default
    for field in fields:
        input_dict[field] = 1

    # Age parsing: match "55 years old", "age is 55", or "age 55"
    age_patterns = [
        r"(\d{2,3})[ -]?(year|yr)[s]?[ -]?old",
        r"age is (\d{2,3})",
        r"age[:\s]+(\d{2,3})"
    ]
    for pattern in age_patterns:
        match = re.search(pattern, q)
        if match:
            input_dict["Age"] = int(match.group(1))
            break  # stop after first match

    # Normalize field names for matching
    field_map = {field.lower(): field for field in fields}

    # Match patterns like "Smoking: 5" or "Coughing of Blood is 7"
    number_pairs = re.findall(r'([A-Za-z\s]+?)(?:[:\s]+is[:\s]*|[:\s]+)([1-9])\b', q)
    for raw_key, val in number_pairs:
        cleaned_key = raw_key.strip().lower()
        if cleaned_key in field_map:
            input_dict[field_map[cleaned_key]] = int(val)
            

    # Match descriptive levels like "Smoking is high"
    for field in fields:
        lower_field = field.lower()
        for level, score in risk_map.items():
            if f"{lower_field} is {level}" in q:
                input_dict[field] = score
            elif f"{level} {lower_field}" in q:
                input_dict[field] = score
            elif f"{lower_field} {level}" in q:
                input_dict[field] = score

    # Predict
    try:
        result = predict_risk(input_dict)
        st.subheader("Prediction")
        st.markdown(f"**Risk Level:** {result['risk_level']}")
        st.markdown(f"**Insight:** {result['insight']}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    st.subheader("Input Sent to Model")
    st.json(input_dict)
