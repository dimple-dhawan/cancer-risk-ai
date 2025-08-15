import pandas as pd
import streamlit as st
from joblib import load
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import re

# --- Setup ---
st.set_page_config(page_title="Cancer Prediction AI", layout="centered")
st.title("🧬 Cancer Prediction AI")

# --- Load Model and Data ---
model = load("cancer_model.joblib")
label_encoder = load("label_encoder.joblib")
df = pd.read_csv("cleaned_cancer_data.csv")

# Create human-readable gender column
if pd.api.types.is_numeric_dtype(df["Gender"]):
    df["GenderLabel"] = df["Gender"].map({1: "Male", 2: "Female"})
else:
    df["GenderLabel"] = (
        df["Gender"].astype(str).str.strip().str.lower()
        .map({"male": "Male", "m": "Male", "female": "Female", "f": "Female"})
    )
df["GenderLabel"] = df["GenderLabel"].fillna("Unknown")

# --- OpenAI Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- Column Mapping to match training data ---
COLUMN_MAPPING = {col.lower(): col for col in model.feature_names_in_}

# --- Encoding qualitative values ---
ENCODING_MAP = {
    "gender": {"male": 0, "female": 1},
    "air pollution": {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 8},
    "alcohol use": {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 8},
    "dust allergy": {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 8},
    "occupational hazards": {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 8},
    "genetic risk": {"low": 1, "medium": 5, "high": 8},
    "chronic lung disease": {"no": 1, "yes": 9, "low": 1, "medium": 5, "high": 8},
    "balanced diet": {"poor": 1, "average": 5, "good": 8, "no": 1, "yes": 8, "low": 1, "medium": 5, "high": 8},
    "obesity": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7, "very high": 8},
    "smoking": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7, "very high": 8},
    "passive smoker": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7, "very high": 8},
    "chest pain": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
    "coughing of blood": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
    "fatigue": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
    "weight loss": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
    "shortness of breath": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
    "wheezing": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
    "swallowing difficulty": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
    "clubbing of finger nails": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
    "frequent cold": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
    "dry cough": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
    "snoring": {"no": 1, "yes": 8, "low": 1, "medium": 5, "high": 7},
}

NUMERIC_COLUMNS = [
    "Air Pollution", "Alcohol Use", "Dust Allergy", "Occupational Hazards",
    "Genetic Risk", "Chronic Lung Disease", "Balanced Diet", "Obesity",
    "Smoking", "Passive Smoker", "Chest Pain", "Coughing of Blood",
    "Fatigue", "Weight Loss", "Shortness of Breath", "Wheezing",
    "Swallowing Difficulty", "Clubbing of Finger Nails", "Frequent Cold",
    "Dry Cough", "Snoring"
]

VALUE_MAPPING = {
    "very low": [1, 2],
    "low": [3, 4],
    "medium": [5, 6],
    "high": [7, 8],
    "very high": [9]
}

# --- GPT Feature Extraction ---
def ask_openai_uncached(user_query, task="predict"):
    if task == "predict":
        prompt = f"""
You are an assistant that extracts patient features from text for cancer risk prediction.
Available features: {', '.join(df.columns)}
Return ONLY a JSON with lowercase keys and values (numbers or descriptors like 'low', 'medium', 'high', 'yes', 'no').
"""
    else:
        prompt = f"""
You are an assistant that converts natural language dataset questions into pandas queries.

The dataset ONLY has the following columns: Age, Gender, Air Pollution, Alcohol Use, Dust Allergy, Occupational Hazards, Genetic Risk, Chronic Lung Disease, Balanced Diet, Obesity, Smoking, Passive Smoker, Chest Pain, Coughing of Blood, Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring, GenderLabel, Level.

All numeric columns use a 1–9 scale: Age, Air Pollution, Alcohol Use, Dust Allergy, Occupational Hazards, Genetic Risk, Chronic Lung Disease, Balanced Diet, Obesity, Smoking, Passive Smoker, Chest Pain, Coughing of Blood, Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring.

The 'Level' column contains only 'High', 'Medium', 'Low'. Map any risk questions to this column. Do not apply numeric operations to 'Level'.

For gender use 'GenderLabel' with 'Male' or 'Female'.

NEVER create, assume, or reference columns not in the list.

Rules for numeric columns on a 1–9 scale:
- Qualitative terms 'very low', 'low', 'medium', 'high', 'very high', 'highly' map to numeric ranges: very low=[1,2], low=[3,4], medium=[5,6], high=[7,8], very high=[9], highly=[8,9].
- If the user mentions a numeric column **without a qualitative term**, assume the high range (7–9).

Always output a valid pandas **filtering expression** using only df[...] syntax.  
Do NOT compute counts or sums. Do NOT use df.count() or df.sum().  
The output should return a filtered DataFrame ready for `.shape[0]` to get the number of matching rows.

Example:  
- "How many obese females?" → `df[(df['Obesity'] > 6) & (df['GenderLabel'] == 'Female')]`  
- "People with high Balanced Diet" → `df[df['Balanced Diet'] >= 7]`
"""




    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

@st.cache_data(show_spinner=False)
def ask_openai(user_query, task="predict"):
    return ask_openai_uncached(user_query, task)

# --- Process input for prediction ---
def process_input(user_input):
    try:
        input_dict = json.loads(user_input)
    except:
        return None
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
    final_input = {col_name: encoded_dict.get(col_lower, 1) for col_lower, col_name in COLUMN_MAPPING.items()}
    return pd.DataFrame([final_input])[model.feature_names_in_]


# Replace qualitative terms with numeric ranges
import re

def convert_qualitative_to_numeric(gpt_query: str) -> str:
    """
    Converts GPT queries with qualitative ranges into safe .isin([...]) format.
    Example: (df['Dust Allergy'] >= 5 & df['Dust Allergy'] <= 6) -> df['Dust Allergy'].isin([5,6])
    """
    pattern = re.compile(r"\(df\['(\w+)'\]\s*>=\s*(\d+)\s*&\s*df\['\1'\]\s*<=\s*(\d+)\)", re.IGNORECASE)


    def replace_range(match):
        col_name = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        nums = list(range(start, end + 1))
        return f"df['{col_name}'].isin({nums})"

    # First, fix ranges
    converted_query = pattern.sub(replace_range, gpt_query)

    # Then, convert any remaining single conditions like df['Col'] >= 7
    pattern_single = re.compile(r"df\['(\w+)'\]\s*>=\s*(\d+)", re.IGNORECASE)
    def replace_single(match):
        col_name = match.group(1)
        start = int(match.group(2))
        nums = list(range(start, 10))  # assuming scale 1–9
        return f"df['{col_name}'].isin({nums})"
    
    converted_query = pattern_single.sub(replace_single, converted_query)

    return converted_query




def query_to_sentence(gpt_query, df):
    # Compute the count
    try:
        allowed = {"df": df, "pd": pd, "np": np}
        result = eval(gpt_query, {"__builtins__": {}}, allowed)
    except Exception as e:
        return f"Error computing result: {e}"

    count = result.shape[0] if isinstance(result, pd.DataFrame) else int(result)

    # Subject (gender or people)
    match = re.search(r"df\['GenderLabel'\]\s*==\s*['\"](\w+)['\"]", gpt_query)
    if match:
        gender = match.group(1).lower()
        if gender == "male":
            subject = "males"
        elif gender == "female":
            subject = "females"
        else:
            subject = "people"
    else:
        subject = "people"

    # Age
    match = re.search(r"df\['Age'\]\s*([><]=?)\s*(\d+)", gpt_query)
    age_part = ""
    if match:
        op, val = match.groups()
        if ">" in op:
            age_part = f"over {val}"
        elif "<" in op:
            age_part = f"under {val}"
        else:
            age_part = f"equal to {val}"

    # Numeric 1–9 columns
    numeric_columns = [
        'Air Pollution', 'Alcohol Use', 'Dust Allergy', 'Occupational Hazards',
        'Genetic Risk', 'Chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',
        'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
        'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
        'Frequent Cold', 'Dry Cough', 'Snoring'
    ]

    # Collect all feature descriptions
    feature_descriptions = []
    ADJECTIVES = ["very low", "low", "medium", "high", "very high", "highly"]


    for col in numeric_columns:
        if re.search(rf"df\['{col}'\]\s*(>=|>|<=|<|==)|df\['{col}'\]\.isin\(\[[\d, ]+\]\)", gpt_query):
            # Handle obesity separately to avoid duplicates
            if col.lower() == "obesity":
                if "obesity" not in feature_descriptions:
                    feature_descriptions.append("obesity")
                continue

            # Check for adjective in the user query
            adjective = ""
            for adj in ADJECTIVES:
                # Match adjective followed by the column name ignoring plural 's'
                pattern = rf"{adj}\s+{col.lower()}s?"
                if re.search(pattern, gpt_query.lower()):
                    adjective = adj.upper() + " "
                    break
            feature_descriptions.append(f"{adjective}{col.lower()}")

    # Level (cancer risk)
    match = re.search(r"df\['Level'\]\s*==\s*['\"](\w+)['\"]", gpt_query)
    if match:
        feature_descriptions.append(f"at {match.group(1).lower()} risk of cancer")

    # Build sentence naturally
    parts = [subject]
    if age_part:
        parts.append(age_part)

    if feature_descriptions:
        # Use natural sentence if 1–3 features, else generic
        if len(feature_descriptions) <= 3:
            # First feature uses "are" or "have" after subject
            sentence = f"{count} {' '.join(parts)} have {feature_descriptions[0]}"
            if len(feature_descriptions) > 1:
                sentence += " and  " + " and ".join(feature_descriptions[1:])
            sentence += " in the dataset."
        else:
            sentence = f"{count} {' '.join(parts)} match the query conditions in the dataset."
    else:
        sentence = f"{count} {' '.join(parts)} are in the dataset."

    return sentence



# --- Predict risk ---
def predict_from_input(input_df):
    try:
        prediction = model.predict(input_df)[0]
        risk_level = label_encoder.inverse_transform([prediction])[0]
        messages = {
            "Low": "Risk appears low. Continue standard monitoring and apply clinical judgment.",
            "Medium": "Risk appears moderate. Consider additional evaluation or monitoring.",
            "High": "Risk appears high. Review patient history and clinical indicators."
        }
        return f"Predicted Cancer Risk: **{risk_level.capitalize()}**\n\n{messages[risk_level]}"
    except Exception as e:
        return f"Error during prediction: {e}"

# --- Streamlit UI ---
st.text("Features include: Age, Gender, Air Pollution, Alcohol Use, Dust Allergy, Occupational Hazards, Genetic Risk, Chronic Lung Disease, Balanced Diet, Obesity, Smoking, Passive Smoker, Chest Pain, Coughing of Blood, Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring")
st.text("Enter patient info or ask a dataset question:")

user_input = st.text_area(
    """Examples:
- '50 year old, male, smoker, medium fatigue' or 'How many men under 30 are high risk of cancer and high obesity?'""",
    height=100
)

if st.button("Submit") and user_input:
    with st.spinner("Analyzing..."):
        # Decide if this is a prediction or dataset query
        if re.search(r"how many|count|number of", user_input.lower()):
            gpt_query = ask_openai(user_input, task="dataset")
            gpt_query = convert_qualitative_to_numeric(gpt_query)  # ensures 'high' -> .isin([7,8])
            try:
                allowed = {"df": df, "pd": pd, "np": np}
                result = eval(gpt_query, {"__builtins__": {}}, allowed)
                sentence = query_to_sentence(gpt_query, result)
                st.success(sentence)

            except Exception as e:
                st.error(f"I did not understand your request.  Please try again.")
        else:
            extracted_json = ask_openai(user_input, task="predict")
            input_df = process_input(extracted_json)
            if input_df is not None:
                result = predict_from_input(input_df)
                st.success(result)
            else:
                st.error("Could not process input. Check your text format.")


# Disclaimer
st.markdown(
    "**Disclaimer:** This tool is for educational purposes only. Not for medical diagnosis or treatment."
)
