import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------------
# Load trained regression model
# ---------------------------------------------------------
MODEL_FILE = "flight_price_linear_regression_dt.pkl"

try:
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(
        f"Model file '{MODEL_FILE}' not found. "
        "Save your trained xgb_best model from the notebook first."
    )
    st.stop()

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Price Prediction")
st.write("Enter your flight details to estimate the ticket price.")

st.divider()

# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------
with st.form("flight_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox(
            "Airline",
            [
                "Air India",
                "AirAsia",
                "GO FIRST",
                "Indigo",
                "SpiceJet",
                "StarAir",
                "Trujet",
                "Vistara"
            ]
        )

        source = st.selectbox(
            "Departure City",
            [
                "Bangalore",
                "Chennai",
                "Delhi",
                "Hyderabad",
                "Kolkata",
                "Mumbai"
            ]
        )

        destination = st.selectbox(
            "Arrival City",
            [
                "Bangalore",
                "Chennai",
                "Delhi",
                "Hyderabad",
                "Kolkata",
                "Mumbai"
            ]
        )

        flight_class = st.selectbox(
            "Class",
            ["Economy", "Business"]
        )

    with col2:
        departure_time = st.slider(
            "Departure Hour",
            min_value=0,
            max_value=23,
            value=10
        )

        arrival_time = st.slider(
            "Arrival Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        stops = st.selectbox(
            "Number of Stops",
            ["Non-stop", "1-stop", "2+-stop"]
        )

        duration_hours = st.number_input(
            "Travel Duration (Hours)",
            min_value=0,
            max_value=50,
            value=2
        )

        duration_minutes = st.selectbox(
            "Additional Minutes",
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        )

    submitted = st.form_submit_button(
        "Predict Flight Price",
        use_container_width=True
    )

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:

    # Convert categorical values into the same numerical
    # representation used in the notebook.
    class_value = 1 if flight_class == "Business" else 0

    stop_value = {
        "Non-stop": 0,
        "1-stop": 1,
        "2+-stop": 2
    }[stops]

    trip_time = (duration_hours * 60) + duration_minutes

    # Base features
    input_data = {
        "dep_time": departure_time,
        "arr_time": arrival_time,
        "Class": class_value,
        "stop_new": stop_value,
        "Trip_Time(min)": trip_time,

        # Airline dummy variables
        "airline_AirAsia": 0,
        "airline_GO FIRST": 0,
        "airline_Indigo": 0,
        "airline_SpiceJet": 0,
        "airline_StarAir": 0,
        "airline_Trujet": 0,
        "airline_Vistara": 0,

        # Source dummy variables
        "from_Chennai": 0,
        "from_Delhi": 0,
        "from_Hyderabad": 0,
        "from_Kolkata": 0,
        "from_Mumbai": 0,

        # Destination dummy variables
        "to_Chennai": 0,
        "to_Delhi": 0,
        "to_Hyderabad": 0,
        "to_Kolkata": 0,
        "to_Mumbai": 0
    }

    # Set selected airline dummy
    if airline != "Air India":
        input_data[f"airline_{airline}"] = 1

    # Set selected source dummy
    if source != "Bangalore":
        input_data[f"from_{source}"] = 1

    # Set selected destination dummy
    if destination != "Bangalore":
        input_data[f"to_{destination}"] = 1

    # Create DataFrame in the exact feature order used by the model
    feature_columns = [
        "dep_time",
        "arr_time",
        "Class",
        "stop_new",
        "Trip_Time(min)",
        "airline_AirAsia",
        "airline_GO FIRST",
        "airline_Indigo",
        "airline_SpiceJet",
        "airline_StarAir",
        "airline_Trujet",
        "airline_Vistara",
        "from_Chennai",
        "from_Delhi",
        "from_Hyderabad",
        "from_Kolkata",
        "from_Mumbai",
        "to_Chennai",
        "to_Delhi",
        "to_Hyderabad",
        "to_Kolkata",
        "to_Mumbai"
    ]

    test_df = pd.DataFrame([input_data])[feature_columns]

    prediction = model.predict(test_df)[0]

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------
    st.divider()

    st.success(
        f"### Estimated Flight Price: ₹{prediction:,.0f}"
    )

    st.caption(
        "The prediction is generated using the trained regression model "
        "from the Flight Price Prediction project."
    )

    with st.expander("View Input Details"):
        display_data = pd.DataFrame({
            "Feature": [
                "Airline",
                "Departure City",
                "Arrival City",
                "Class",
                "Departure Hour",
                "Arrival Hour",
                "Stops",
                "Travel Duration"
            ],
            "Value": [
                airline,
                source,
                destination,
                flight_class,
                departure_time,
                arrival_time,
                stops,
                f"{duration_hours}h {duration_minutes}m"
            ]
        })

        st.dataframe(display_data, use_container_width=True, hide_index=True)
