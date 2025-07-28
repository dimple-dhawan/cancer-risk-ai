import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Load the dataset
df = pd.read_csv("cancer_data.csv")

# Convert categorical variables
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Gender to numeric
label_encoder = LabelEncoder()
df['Level'] = label_encoder.fit_transform(df['Level'])  # Target to numeric

# Split features and target
X = df.drop("Level", axis=1)
y = df["Level"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
dump(model, "cancer_model.joblib")

# Save the label encoder (so you can decode predictions later)
dump(label_encoder, "label_encoder.joblib")

print("Model trained and saved as cancer_model.joblib")
