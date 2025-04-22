import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Define the ModelInference class
class ModelInference:
    def __init__(self, model_path):
        # Load the model
        self.model = joblib.load(model_path)  # Loading the model
        if not hasattr(self.model, 'predict'):
            raise ValueError("The loaded object is not a valid model with a 'predict' method.")
        
    def predict(self, input_data):
        # Ensure the input data is in a DataFrame format
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])  # Convert dict to DataFrame
        
        # Handle missing values by filling them with zeros (or use mean/median)
        if input_data.isnull().values.any():
            st.warning("Input data contains missing values. Filling with zeros.")
            input_data = input_data.fillna(0)  # Or use .fillna(input_data.mean()) if needed
        
        # Ensure correct data types (all features must be numeric)
        for col in input_data.columns:
            if not pd.api.types.is_numeric_dtype(input_data[col]):
                raise ValueError(f"Feature {col} has invalid data type. Expected numeric data.")
        
        # Make prediction with the model
        prediction = self.model.predict(input_data)
        return prediction

# Load the trained model with ModelInference
model_inference = ModelInference("rf_model.pkl")

def main():
    st.title("Hotel Booking Cancellation Prediction")

    # Input fields from the user
    no_of_adults = st.number_input("Number of Adults", min_value=0, value=2)
    no_of_children = st.number_input("Number of Children", min_value=0, value=0)
    no_of_guests = st.number_input("Total Guests", min_value=1, value=2)
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)
    type_of_meal_plan = st.number_input("Meal Plan", min_value=1, max_value=5, value=1) 
    required_car_parking_space = st.number_input("Car Parking", min_value=0, max_value=1, value=0)
    room_type_reserved = st.number_input("Room Type", min_value=1, max_value=7, value=1)
    lead_time = st.number_input("Lead Time", min_value=0, value=224)
    arrival_year = st.number_input("Arrival Year", min_value=0, value=2017)
    arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=10)
    arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, value=2)
    market_segment_type = st.number_input("Market Segment", min_value=1, max_value=5, value=2)
    repeated_guest = st.number_input("Is Repeated Guest", min_value=0, max_value=1, value=0)
    no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Prev Bookings Not Canceled", min_value=0, value=0)
    avg_price_per_room = st.number_input("Average Price", min_value=0.0, value=65.00)
    no_of_special_requests = st.number_input("Special Requests", min_value=0, value=0)

    # Prediction button
    if st.button("Predict Cancellation"):
        # Prepare input data as a dictionary
        features = {
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_guests": no_of_guests,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal_plan": type_of_meal_plan,
            "required_car_parking_space": required_car_parking_space,
            "room_type_reserved": room_type_reserved,
            "lead_time": lead_time,
            "arrival_year": arrival_year,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "market_segment_type": market_segment_type,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests,
        }

        # Make prediction using the ModelInference class
        try:
            prediction = model_inference.predict(features)
            # Display result
            st.success(f"Prediction: {'Canceled' if prediction[0] == 1 else 'Not Canceled'}")
        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as ex:
            st.error(f"An unexpected error occurred: {ex}")

if __name__ == "__main__":
    main()
