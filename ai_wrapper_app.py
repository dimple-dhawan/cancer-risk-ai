import pandas as pd
import streamlit as st
import openai
from joblib import load
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

# --- Setup ---
st.set_page_config(page_title="Cancer Prediction AI App", layout="centered")
st.title("🧬 Cancer Prediction AI App")

# --- Load Model, Encoder, and Data ---
model = load("cancer_model.joblib")
label_encoder = load("label_encoder.joblib")
df = pd.read_csv("cleaned_cancer_data.csv")

# Set your OpenAI API Key
load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded? {'Yes' if api_key else 'No'}")
client = OpenAI(api_key=api_key)

# --- Define Function to Interact with OpenAI ---
def ask_openai(user_query):
    prompt = f"""
You are an AI analyst helping users query a cancer dataset. The dataset includes patient features like age, gender, smoking, obesity, etc.

Here are the columns in the dataset:
{', '.join(df.columns)}

The user may:
1. Ask a question like "how many", "how much", "what is the average", etc. In that case, return Python pandas code that computes it using the DataFrame called df.
    - Example: "How many smokers are in the dataset?"
    - Your output should be Python code only, with a comment describing what it's doing.

2. Provide natural input asking for a risk prediction — for example:
    - "What is the risk for a 30-year-old obese male smoker?"
    - Or shorthand like: "30 year old, male, smoker, obese"
    
In that case:
- Extract the necessary features into a cleaned dictionary using the appropriate column names.
- Convert it to a DataFrame, run model.predict() to get the predicted risk level.
- Return Python code that prints: "The risk level is [low/medium/high] based on the user's input."

Return only Python code in your response.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def fill_missing_features(input_dict, df):
    filled = input_dict.copy()
    for col in df.columns:
        if col not in filled:
            filled[col] = 1
    return filled

# --- Helper to Run Predictions ---
def predict_from_input(input_dict):
    try:
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]  # numeric from 1 to 9

        # Map numeric to risk category
        if prediction <= 3:
            risk_level = "low"
        elif prediction <= 6:
            risk_level = "medium"
        else:
            risk_level = "high"

        messages = {
            "low": "Patient is at low risk. Continue monitoring and routine follow-up.",
            "medium": "Patient shows moderate risk. Consider checking for early signs of non-adherence.",
            "high": "Patient is at high risk. Immediate intervention may be needed to prevent complications."
        }

        message = messages.get(risk_level, "Risk level unknown.")

        return f"Predicted Cancer Risk Level: **{risk_level.capitalize()}**\n\n{message}"
    except Exception as e:
        return f"Error during prediction: {e}"

st.text("Features include:  Age, Gender, Air Pollution, Alcohol use, Dust Allergy, OccuPational Hazards, Genetic Risk, Chronic Lung Disease, Balanced Diet, Obesity, Smoking, Passive Smoker, Chest Pain, Coughing of Blood, Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring")

# --- User Input ---
user_question = st.text_area("Ask me a question about the cancer data or a patient case:", height=100)

if st.button("Submit") and user_question:
    with st.spinner("Analyzing..."):
        response = ask_openai(user_question)

    st.markdown("### 🤖 OpenAI Response")
    st.code(response)

    # Try to parse and execute result
    try:
        if "predict" in response.lower() or "model" in response.lower():
            # Attempt to extract dictionary
            exec_globals = {}
            exec(response, {"np": np}, exec_globals)
            input_dict = exec_globals.get("input_dict")
            if input_dict:
                input_dict_filled = fill_missing_features(input_dict, df)
                result = predict_from_input(input_dict_filled)
                st.success(result)
            else:
                st.error("No input_dict found in response.")
        elif "df" in response:
            exec_globals = {"df": df.copy()}
            exec(response, exec_globals)
            result = exec_globals.get("result")
            if result is not None:
                st.dataframe(result)
            else:
                st.warning("No 'result' variable in code. Showing output of execution.")
        else:
            st.warning("Unrecognized format. Manual response:\n" + response)
    except Exception as e:
        st.error(f"Error executing OpenAI response:\n{e}")
