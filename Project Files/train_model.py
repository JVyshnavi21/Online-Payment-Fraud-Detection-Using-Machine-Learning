import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("🔄 Loading dataset...")
df = pd.read_csv("fraud.csv")

# ---------------- SAMPLE DATA ----------------
df = df.sample(n=100000, random_state=42)  # use 100k rows only for fast training

# ---------------- CLEAN DATA ----------------
df = df.drop(['nameOrig', 'nameDest'], axis=1, errors='ignore')

# Encode transaction type
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# ---------------- FEATURES ----------------
features = ['type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
X = df[features]
y = df['isFraud']

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=50,  # fewer trees for speed
    n_jobs=-1,        # use all CPU cores
    random_state=42
)

print("🚀 Training model on 100k rows...")
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Model accuracy: {accuracy:.4f}")

# ---------------- SAVE MODEL ----------------
with open("model.pkl", "wb") as f:
    pickle.dump((model, le), f)

print("🎉 model.pkl saved successfully!")