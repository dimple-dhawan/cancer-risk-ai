import pandas as pd
from joblib import load

# Load the model and label encoder
model = load("cancer_model.joblib")
label_encoder = load("label_encoder.joblib")

# Sample input data (must match feature names exactly as trained)
input_data = pd.DataFrame([{
    'Age': 55,
    'Gender': 0,  # 0 = Male, 1 = Female
    'Air Pollution': 1,
    'Alcohol use': 1,
    'Dust Allergy': 1,
    'OccuPational Hazards': 1,
    'Genetic Risk': 1,
    'Chronic Lung Disease': 1,  # Make sure case matches training
    'Balanced Diet': 1,
    'Obesity': 7,
    'Smoking': 7,
    'Passive Smoker': 7,
    'Chest Pain': 1,
    'Coughing of Blood': 7,
    'Fatigue': 7,
    'Weight Loss': 1,
    'Shortness of Breath': 7,
    'Wheezing': 7,
    'Swallowing Difficulty': 7,
    'Clubbing of Finger Nails': 1,
    'Frequent Cold': 3,
    'Dry Cough': 3,
    'Snoring': 1
}])

# Make prediction
prediction = model.predict(input_data)
prediction_label = label_encoder.inverse_transform(prediction)[0]

# Generate insight
def generate_insight(risk_level):
    if risk_level == "Low":
        return "Patient is at low risk. Continue monitoring and routine follow-up."
    elif risk_level == "Medium":
        return "Patient shows moderate risk. Consider checking for early signs of non-adherence."
    elif risk_level == "High":
        return "Patient is at high risk. Immediate intervention may be needed to prevent complications."
    else:
        return "Unknown risk level. Please review input data."

insight = generate_insight(prediction_label)

# Output
print("Predicted Cancer Risk Level:", prediction_label)
print("AI Insight:", insight)
