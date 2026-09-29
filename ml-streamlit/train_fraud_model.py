import json
from pathlib import Path
 
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
 
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = "dataset-carclaims.csv"  
 
FEATURES = [
    "Make",
    "VehicleCategory",
    "AgeOfVehicle",
    "VehiclePrice",
    "PastNumberOfClaims",
    "AgeOfPolicyHolder",
    "NumberOfSuppliments",
    "PolicyType",
    "AccidentArea",
]
TARGET = "FraudFound"
 
df = pd.read_csv(DATA_PATH)
 
X_raw = df[FEATURES].astype(str)
X = pd.get_dummies(X_raw).astype(int)
y = df[TARGET].map({"No": 0, "Yes": 1})
assert y.notna().all(), f"unexpected values in {TARGET}: {df[TARGET].unique()}"
 
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
 
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)
model.fit(X_train, y_train)
 
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.3f}")
print(classification_report(y_test, pred, target_names=["not fraud", "fraud"]))
 
joblib.dump(model, BASE_DIR / "fraud_model.pkl")
joblib.dump(list(X.columns), BASE_DIR / "feature_columns.pkl")
categories = {c: sorted(X_raw[c].unique().tolist()) for c in FEATURES}
(BASE_DIR / "categories.json").write_text(json.dumps(categories, indent=2))
 
print("saved fraud_model.pkl, feature_columns.pkl, categories.json")
