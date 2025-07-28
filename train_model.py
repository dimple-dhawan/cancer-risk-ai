import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("original_cancer_patient_data_sets.csv")

# 2. Clean and encode categorical values
df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize().map({'Male': 0, 'Female': 1})

# Rename column to fix casing (optional cleanup)
df.rename(columns={"chronic Lung Disease": "Chronic Lung Disease"}, inplace=True)

label_encoder = LabelEncoder()
df['Level'] = label_encoder.fit_transform(df['Level'])

# 3. Drop non-feature columns
X = df.drop(columns=["Patient Id", "Level"])
X = X.loc[:, ~X.columns.str.contains('^index$', case=False)]  # This removes 'index
y = df["Level"]

# 4. Show class distribution
print("Class distribution:")
print(df['Level'].value_counts())

# 5. Check for missing values
print("Missing values:\n", df.isnull().sum())

# 6. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train model
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Feature importance
importances = model.feature_importances_
feat_names = X.columns
sorted_idx = importances.argsort()

plt.figure(figsize=(10, 8))
plt.barh(feat_names[sorted_idx], importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# 10. Save model and encoder
dump(model, "cancer_model.joblib")
dump(label_encoder, "label_encoder.joblib")
print("\n✅ Model and label encoder saved.")

print(model.feature_names_in_)
