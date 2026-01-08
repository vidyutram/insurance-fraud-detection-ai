import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("dataset-carclaims(1).csv")

# Features and target
features = [
    "Make",
    "VehicleCategory",
    "AgeOfVehicle",
    "VehiclePrice",
    "PastNumberOfClaims",
    "AgeOfPolicyHolder",
    "NumberOfSuppliments",
    "PolicyType",
    "AccidentArea"
]

target = "FraudFound"

X = pd.get_dummies(df[features])
y = df[target].map({"No": 0, "Yes": 1})

# Handle imbalance
scale_pos_weight = sum(y == 0) / sum(y == 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))

# Save model and feature columns
joblib.dump(model, "fraud_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("Model and feature columns saved.")
