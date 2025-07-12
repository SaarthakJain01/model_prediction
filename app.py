import streamlit as st
import pandas as pd
import joblib  # or use pickle
import numpy as np

# Load pipeline
loaded_pipeline = joblib.load("logistic_pipeline.joblib")

# Set page config
st.set_page_config(
    page_title="IPO Application Suggestion",
    page_icon="📈",
    layout="centered"
)

# Title and Intro
st.title("📈 IPO Application Suggestion Tool")

st.markdown("""
This tool uses **historical Mainboard IPO data** and a **Logistic Regression model** to predict whether applying for an IPO is likely to yield **listing gains of more than 5%**.

- ✅ Model Accuracy: **74%** on test data
- 📊 Based on factors like **Issue Price**, **IPO Size**, and **QIB Subscription Level**
- 💼 For educational and informational purposes only
""")

st.divider()

# Input form
st.subheader("Enter IPO Details")
with st.form("input_form"):
    issue_price = st.number_input("Issue Price (₹)", min_value=0.0, step=0.01, help="The fixed or price band's upper end for the IPO.")
    size_in_cr = st.number_input("IPO Size (in ₹ Crores)", min_value=0.0, step=0.01, help="Total issue size of the IPO.")
    qib = st.number_input("QIB Subscription (x)", min_value=0.0, step=0.01, help="Qualified Institutional Buyers (QIB) subscription level in times.")

    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = pd.DataFrame([[issue_price, size_in_cr, qib]], columns=["issue_price", "size_in_cr", "qib"])
    prediction = loaded_pipeline.predict(input_data)[0]

    st.subheader("💡 Prediction Result")
    if prediction == 1:
        st.success("Yes — The IPO is likely to give listing gains.")
    else:
        st.error("No — The IPO may not offer listing gains over 5%.")

    st.caption("Disclaimer: Predictions are probabilistic based on past patterns and not guaranteed outcomes.")

st.divider()

# Sample Info (Optional for transparency)
st.subheader("📘 Example Scenario")
st.markdown("""
Here’s a typical input that resulted in a “Yes” prediction:

- Issue Price: ₹100
- Size: ₹500 Cr
- QIB Subscription: 100x

Model output: ✅ Yes
""")

st.divider()

# About and Footer
st.markdown("""
### 📌 Notes:
- **Model:** Logistic Regression
- **Target:** Listing gain > 5%
- **Data Source:** [Chittorgarh IPO Data](https://www.chittorgarh.com/)
- **Disclaimer:** This is not financial advice. Please do your own research or consult a financial advisor before making investment decisions.

---

👨‍💻 Built by [Saarthak Jain](https://saarthakjain.vercel.app/)  
🔗 [GitHub](https://github.com/SaarthakJain01) | [LinkedIn](https://www.linkedin.com/in/saarthakjain01)
""")
