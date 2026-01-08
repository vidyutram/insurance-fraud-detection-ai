import streamlit as st
from fraud_detection import predict_from_text

st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")

st.title("🛡️ Insurance Fraud Detection App")
st.write("Enter claim details below and detect whether it is **Fraud** or **Not Fraud**.")

user_input = st.text_area(
    "Enter claim description",
    placeholder="Type your claim details here..."
)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to predict.")
    else:
        # 🔹 CALL MODEL
        pred, details, fraud_prob = predict_from_text(user_input)

        if details is None:
            st.error("❌ Could not extract claim details.")
        else:
            # 🔹 SHOW EXTRACTED DETAILS
            st.subheader("📄 Extracted Details")
            st.json(details)

            # 🔹 SHOW PREDICTION
            st.subheader("🔍 Prediction Result")
            st.write(f"Fraud probability: **{fraud_prob:.2f}**")

            if pred == 1:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ NOT FRAUD")
                st.caption("⚠️ Fraud threshold set at 0.75 for reduced false positives")

