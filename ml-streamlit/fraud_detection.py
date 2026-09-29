import difflib
import json
import os
import re
from pathlib import Path
 
import joblib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
 
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / "backend" / ".env")
 
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
 
THRESHOLD = 0.5
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
EXTRACTOR = os.getenv("EXTRACTOR", "openai")  # "openai" or "langchain"
 
model = joblib.load(BASE_DIR / "fraud_model.pkl")
feature_cols = joblib.load(BASE_DIR / "feature_columns.pkl")
categories = json.loads((BASE_DIR / "categories.json").read_text())
 
 
def _api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (see backend/.env.example)")
    return key
 

def build_instructions():
    lines = [
        "You extract insurance claim details from free text.",
        "Return ONLY a JSON object with exactly the keys below.",
        "Each value must be copied exactly from its allowed list. "
        "If the text does not give enough info for a field, use null.",
        "For numbers (age, price, claims), pick the allowed range that contains the number.",
        "",
    ]
    for col in FEATURES:
        lines.append(f"{col}: {json.dumps(categories[col])}")
    return "\n".join(lines)
 
 
def _parse_json(content):
    match = re.search(r"\{.*\}", content or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None
 
 
def _extract_openai(text):
    client = OpenAI(base_url=OPENROUTER_URL, api_key=_api_key())
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": build_instructions()},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return _parse_json(res.choices[0].message.content)
 
 
def _extract_langchain(text):
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
 
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=OPENROUTER_URL,
        api_key=_api_key(),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{instructions}"), ("human", "{text}")]
    )
    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({"instructions": build_instructions(), "text": text})
 
 
def extract_attributes(text):
    try:
        if EXTRACTOR == "langchain":
            return _extract_langchain(text)
        return _extract_openai(text)
    except Exception as e:
        print("extraction error:", e)
        return None
 
 
def normalise(raw):
    clean, missing = {}, []
    for col in FEATURES:
        allowed = {a.lower(): a for a in categories[col]}
        val = raw.get(col)
        match = None
        if val is not None and str(val).strip():
            v = str(val).strip().lower()
            if v in allowed:
                match = allowed[v]
            else:
                close = difflib.get_close_matches(v, list(allowed), n=1, cutoff=0.8)
                if close:
                    match = allowed[close[0]]
        clean[col] = match
        if match is None:
            missing.append(col)
    return clean, missing
 
 
def predict_from_text(text):
    raw = extract_attributes(text)
    if not raw:
        return None, None, None, []
 
    details, missing = normalise(raw)
 
    row = pd.DataFrame(0, index=[0], columns=feature_cols)
    for col, val in details.items():
        name = f"{col}_{val}"
        if val is not None and name in row.columns:
            row.at[0, name] = 1
 
    fraud_prob = float(model.predict_proba(row)[0][1])
    pred = int(fraud_prob >= THRESHOLD)
    return pred, details, fraud_prob, missing
