import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =============================
# Load dataset
# =============================
df = pd.read_csv("fraud_dataset_1000.csv")

# =============================
# Encode 'type'
# =============================
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])

# =============================
# Features and target
# =============================
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Save feature order
features = X.columns.tolist()

# =============================
# Train model
# =============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# =============================
# Save artifacts (IMPORTANT)
# =============================
with open("model.pkl", "wb") as f:
    pickle.dump((model, le, features), f)

print("✅ Model trained and saved successfully!")
