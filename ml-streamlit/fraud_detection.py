import re
import json
import joblib
import pandas as pd
import os
from openai import OpenAI

# ✅ Correct OpenRouter + OpenAI SDK setup
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Load trained ML artifacts
model = joblib.load("fraud_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")


def extract_attributes(text):
    prompt = f"""
Extract the following fields in STRICT JSON format.
If a value is missing, use a reasonable default.

Fields:
- Make
- VehicleCategory
- AgeOfVehicle
- VehiclePrice
- PastNumberOfClaims
- AgeOfPolicyHolder
- NumberOfSupplements
- PolicyType
- AccidentArea

Text:
{text}
"""

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content
    match = re.search(r"\{.*\}", content, re.DOTALL)

    return json.loads(match.group()) if match else {}


def predict_from_text(text):
    details = extract_attributes(text)
    if not details:
        return None, None, None

    # Convert extracted JSON to DataFrame
    df_input = pd.DataFrame([details])

    # One-hot encode + align with training columns
    df_enc = pd.get_dummies(df_input)
    df_enc = df_enc.reindex(columns=feature_cols, fill_value=0)

    # Get fraud probability
    fraud_prob = model.predict_proba(df_enc)[0][1]

    # Threshold-based decision
    threshold = 0.75
    pred = 1 if fraud_prob >= threshold else 0

    return pred, details, fraud_prob


