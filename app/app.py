from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction")
st.write("Enter the customer details and calculate the churn risk.")


# Load the saved model
model_path = (
    Path(__file__).resolve().parent
    / "model"
    / "final_xgboost_pipeline.joblib"
)

try:
    model = joblib.load(model_path)
except Exception as error:
    st.error(f"Could not load the saved model: {error}")
    st.stop()


THRESHOLD = 0.35


# Customer details
st.subheader("Customer details")

left, right = st.columns(2)

with left:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_label = st.selectbox(
        "Senior citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Has a partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Has dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure in months",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone service",
        ["Yes", "No"]
    )

    if phone_service == "Yes":
        multiple_lines = st.selectbox(
            "Multiple phone lines",
            ["No", "Yes"]
        )
    else:
        multiple_lines = "No phone service"
        st.caption("Multiple lines set to: No phone service")


with right:
    internet_service = st.selectbox(
        "Internet service",
        ["DSL", "Fiber optic", "No"]
    )

    internet_options = [
        "Online security",
        "Online backup",
        "Device protection",
        "Technical support",
        "Streaming television",
        "Streaming movies"
    ]

    if internet_service == "No":
        selected_services = []

        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

        st.caption(
            "Internet add-on services set to: No internet service"
        )

    else:
        selected_services = st.multiselect(
            "Internet services used",
            internet_options
        )

        online_security = (
            "Yes" if "Online security" in selected_services else "No"
        )

        online_backup = (
            "Yes" if "Online backup" in selected_services else "No"
        )

        device_protection = (
            "Yes" if "Device protection" in selected_services else "No"
        )

        tech_support = (
            "Yes" if "Technical support" in selected_services else "No"
        )

        streaming_tv = (
            "Yes" if "Streaming television" in selected_services else "No"
        )

        streaming_movies = (
            "Yes" if "Streaming movies" in selected_services else "No"
        )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )


# Calculate total charges automatically
total_charges = float(tenure) * float(monthly_charges)

st.write(
    f"Calculated total charges: **{total_charges:,.2f}**"
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button(
    "Predict churn risk",
    type="primary",
    use_container_width=True
):

    senior_citizen = 1 if senior_label == "Yes" else 0

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [int(tenure)],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [float(monthly_charges)],
        "TotalCharges": [float(total_charges)]
    })

    try:
        churn_probability = float(
            model.predict_proba(customer)[0, 1]
        )

        churn_percentage = churn_probability * 100

        st.subheader("Prediction result")

        st.metric(
            "Estimated churn probability",
            f"{churn_percentage:.2f}%"
        )

        st.progress(churn_probability)

        if churn_probability >= THRESHOLD:
            st.error("Higher churn risk")
        else:
            st.success("Lower churn risk")

        st.caption(
            "Customers with a predicted churn probability of 35% "
            "or higher are classified as higher churn risk."
        )

        with st.expander("View data used for prediction"):
            st.dataframe(
                customer,
                use_container_width=True,
                hide_index=True
            )

            st.write(
                "Number of model inputs:",
                customer.shape[1]
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

