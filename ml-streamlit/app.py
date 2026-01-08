import streamlit as st
from fraud_detection import predict_from_text

st.set_page_config(
    page_title="Insurance Fraud Detection",
    layout="centered"
)

st.title("🛡️ Insurance Fraud Detection App")
st.write(
    "Enter claim details below and detect whether it is **Fraud** or **Not Fraud**."
)

user_input = st.text_area(
    "Enter claim description",
    placeholder="Type your claim details here..."
)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to predict.")
    else:
        with st.spinner("Analyzing claim with AI..."):
            pred, details, fraud_prob = predict_from_text(user_input)

        if details:
            st.subheader("📄 Extracted Details")
            st.json(details)

            st.subheader("🔍 Prediction Result")
            st.write(f"Fraud probability: **{fraud_prob:.2f}**")

            if pred == 1:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ NOT FRAUD")
        else:
            st.error("Could not extract claim details.")
