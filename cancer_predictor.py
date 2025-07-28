import pandas as pd
from joblib import load

# Load model and label encoder once
model = load("cancer_model.joblib")
label_encoder = load("label_encoder.joblib")

def predict_risk(input_dict):
    # Convert Gender string to numeric if needed
    if 'Gender' in input_dict and isinstance(input_dict['Gender'], str):
        input_dict['Gender'] = 0 if input_dict['Gender'].lower() == 'male' else 1

    # Create a DataFrame from input
    input_df = pd.DataFrame([input_dict])

    # Make prediction
    prediction = model.predict(input_df)
    label = label_encoder.inverse_transform(prediction)[0]

    # Insight wrapper
    def generate_insight(risk_level):
        if risk_level == "Low":
            return "Patient is at low risk. Continue monitoring and routine follow-up."
        elif risk_level == "Medium":
            return "Patient shows moderate risk. Consider checking for early signs of non-adherence."
        elif risk_level == "High":
            return "Patient is at high risk. Immediate intervention may be needed to prevent complications."
        return "Risk level unknown."

    return {
        "risk_level": label,
        "insight": generate_insight(label)
    }
