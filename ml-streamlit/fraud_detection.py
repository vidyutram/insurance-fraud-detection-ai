import os
import re
import json
import joblib
import pandas as pd
from openai import OpenAI

# ---------- PATH SAFETY ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model="openai/gpt-3.5-turbo",
model = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

# ---------- OPENROUTER CLIENT ----------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# ---------- ATTRIBUTE EXTRACTION ----------
def extract_attributes(text):
    prompt = f"""
Extract the following fields in JSON format.
If a value is missing, use reasonable defaults.

Fields:
Make,
VehicleCategory,
AgeOfVehicle,
VehiclePrice,
PastNumberOfClaims,
AgeOfPolicyHolder,
NumberOfSupplements,
PolicyType,
AccidentArea

Text:
{text}
"""

    res = client.chat.completions.create(
    model="openai/gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0)

    match = re.search(r"\{.*\}", res.choices[0].message.content, re.DOTALL)

    if not match:
        return None

    return json.loads(match.group())

# ---------- MAIN PREDICTION FUNCTION ----------
def predict_from_text(text):
    details = extract_attributes(text)
    if not details:
        return None, None, None

    df_input = pd.DataFrame([details])
    df_enc = pd.get_dummies(df_input).reindex(
        columns=feature_cols,
        fill_value=0
    )

    pred = model.predict(df_enc)[0]
    fraud_prob = model.predict_proba(df_enc)[0][1]

    return pred, details, fraud_prob
