import pandas as pd
import streamlit as st
from joblib import load
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# --- Setup ---
st.set_page_config(page_title="Cancer Prediction AI", layout="centered")
st.title("🧬 Cancer Prediction AI")

# --- Load Model and Data ---
model = load("cancer_model.joblib")
label_encoder = load("label_encoder.joblib")
df = pd.read_csv("cleaned_cancer_data.csv")

# --- OpenAI API Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- Column Mapping to match training data ---
COLUMN_MAPPING = {col.lower(): col for col in model.feature_names_in_}

# --- Encoding qualitative values to numeric codes (1-9 scale) ---
ENCODING_MAP = {
    "gender": {"male": 0, "female": 1},
    "air pollution":            {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "alcohol use":              {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "dust allergy":             {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "occupational hazards":     {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "genetic risk":             {"low": 1, "medium": 5, "high": 9},
    "chronic lung disease":     {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "balanced diet":            {"poor": 1, "average": 5, "good": 9, "no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "obesity":                  {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "smoking":                  {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "passive smoker":           {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "chest pain":               {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "coughing of blood":        {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "fatigue":                  {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "weight loss":              {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "shortness of breath":      {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "wheezing":                 {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "swallowing difficulty":    {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "clubbing of finger nails": {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "frequent cold":            {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "dry cough":                {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
    "snoring":                  {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 9},
}

# --- GPT Feature Extraction ---
def ask_openai_uncached(user_query):
    prompt = f"""
You are an assistant that extracts patient features from text for cancer risk prediction.

Available features: {', '.join(df.columns)}

User may provide input like:
- "70 year old obese smoker"
- "Smoking: high, 70 year old, obese medium"

You must:
1. Extract all relevant features from the text.
2. Return ONLY a clean JSON object (dictionary) with keys in lowercase, like 'age', 'smoking', 'obesity', etc.
3. Values can be numbers or qualitative descriptors like 'low', 'medium', 'high', 'yes', 'no'.
4. Do not include any explanation, code, or extra text.

If the text contains 'male' or 'female', set the key 'gender' accordingly.

Example output:
{{"age": 70, "smoking": "high", "obesity": "medium"}}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# Cached wrapper to avoid showing "Running ask_openai"
@st.cache_data(show_spinner=False)
def ask_openai(user_query):
    return ask_openai_uncached(user_query)

# --- Process input and map to model columns ---
def process_input(user_input):
    try:
        input_dict = json.loads(user_input)
    except Exception:
        return None

    # Encode qualitative features
    encoded_dict = {}
    for k, v in input_dict.items():
        key_lower = k.lower()
        if key_lower in ENCODING_MAP:
            val_lower = str(v).lower()
            encoded_dict[key_lower] = ENCODING_MAP[key_lower].get(val_lower, 1)
        else:
            try:
                encoded_dict[key_lower] = float(v)
            except:
                encoded_dict[key_lower] = 1

    # Map to model's expected columns
    final_input = {}
    for col_lower, col_name in COLUMN_MAPPING.items():
        final_input[col_name] = encoded_dict.get(col_lower, 1)

    # Create DataFrame with exact same order and names as model expects
    return pd.DataFrame([final_input])[model.feature_names_in_]



# --- Predict risk ---
def predict_from_input(input_df):
    try:
        prediction = model.predict(input_df)[0]  # Directly use DataFrame with correct cols
        risk_level = label_encoder.inverse_transform([prediction])[0]  # Low, Medium, High

        messages = {
            "Low": "Patient is at low risk. Continue monitoring and routine follow-up.",
            "Medium": "Patient shows moderate risk. Consider checking for early signs of non-adherence.",
            "High": "Patient is at high risk. Immediate intervention may be needed to prevent complications."
        }

        return f"Predicted Cancer Risk Level: **{risk_level.capitalize()}**\n\n{messages[risk_level]}"
    except Exception as e:
        return f"Error during prediction: {e}"


# --- Streamlit Interface ---
st.text("Features include: \nAge, Gender, Air Pollution, Alcohol use, Dust Allergy, OccuPational Hazards, Genetic Risk, Chronic Lung Disease, Balanced Diet, Obesity, Smoking, Passive Smoker, Chest Pain, Coughing of Blood, Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring")
st.text("Enter patient information: ")
user_question = st.text_area(
    """If a feature is at a high risk, type the feature name alone.
For other levels, specify the value low, or medium.  \n
ie: '50 year old, male, smoker, fatigue medium'.""",
    height=100
)

if st.button("Submit") and user_question:
    with st.spinner("Analyzing..."):
        extracted_json = ask_openai(user_question)
        input_df = process_input(extracted_json)

    if input_df is not None:
        result = predict_from_input(input_df)  # Pass DataFrame directly
        st.success(result)
    else:
        st.error("Could not process input. Please check your text format.")

