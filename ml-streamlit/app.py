import streamlit as st
 
from fraud_detection import THRESHOLD, predict_from_text
 
st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")
 
st.title("🛡️ Insurance Fraud Detection App")
st.write("Enter claim details below and detect whether it is **Fraud** or **Not Fraud**.")
 
user_input = st.text_area(
    "Enter claim description",
    placeholder="e.g. A Toyota sedan, 2 years old, worth 12000, no past claims, driver age 45, "
    "no supplements, sedan collision policy, accident in a rural area.",
)
 
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to predict.")
    else:
        with st.spinner("Analyzing claim with AI..."):
            pred, details, fraud_prob, missing = predict_from_text(user_input)
 
        if details:
            st.subheader("📄 Extracted Details")
            st.json(details)
 
            if missing:
                st.warning(
                    "Couldn't find or match: " + ", ".join(missing)
                    + ". The prediction is less reliable without them."
                )
 
            st.subheader("🔍 Prediction Result")
            st.write(f"Fraud probability: **{fraud_prob:.2f}** (threshold {THRESHOLD})")
 
            if pred == 1:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ NOT FRAUD")
        else:
            st.error("Could not extract claim details.")
