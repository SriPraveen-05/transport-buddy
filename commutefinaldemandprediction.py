import pandas as pd
import numpy as np
import streamlit as st
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Load the dataset containing bus stops and hourly passenger counts
data = pd.read_csv("D://hackhustle//bus_stops_with_passenger_counts.csv")
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# Streamlit: User input for date range
st.title('Commuter Count Prediction')
start_date = st.date_input("Start Date", value=data['date'].min())
end_date = st.date_input("End Date", value=data['date'].max())

# Filter the data for the selected date range
filtered_data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]

# Function to generate forecasts for a bus stop
def forecast_bus_stop(bus_stop):
    bus_stop_data = filtered_data[filtered_data['bus_stop'] == bus_stop]
    daily_passenger_counts = bus_stop_data.groupby('date').sum().filter(regex='passenger_count_hour').sum(axis=1)
    daily_passenger_counts = daily_passenger_counts.fillna(0)

    # Use AutoARIMA to find the best parameters
    model = auto_arima(daily_passenger_counts, seasonal=True, m=7, trace=True, error_action='ignore', suppress_warnings=True)
    
    # Forecast for the next 7 days
    forecast = model.predict(n_periods=7)
    forecast_dates = pd.date_range(start=end_date, periods=7, freq='D')
    
    return bus_stop, forecast, forecast_dates

# Use ThreadPoolExecutor to parallelize the forecasting process
with ThreadPoolExecutor() as executor:
    forecasts = list(executor.map(forecast_bus_stop, filtered_data['bus_stop'].unique()[:3]))

# Set up dictionary to store forecasts
bus_stop_forecasts = {bus_stop: {'forecast_mean': forecast, 'forecast_dates': forecast_dates} for bus_stop, forecast, forecast_dates in forecasts}

# Initialize variables for max and min forecasts
max_passenger_count_stop = None
min_passenger_count_stop = None
max_passenger_count = -np.inf
min_passenger_count = np.inf

# Loop to find the maximum and minimum forecasts independently
for bus_stop, forecast_data in bus_stop_forecasts.items():
    forecast_mean = forecast_data["forecast_mean"]

    # Check for max forecast
    forecast_max = forecast_mean.max()
    if forecast_max > max_passenger_count:
        max_passenger_count = forecast_max
        max_passenger_count_stop = bus_stop

    # Check for min forecast
    forecast_min = forecast_mean.min()
    if forecast_min < min_passenger_count:
        min_passenger_count = forecast_min
        min_passenger_count_stop = bus_stop

# Display the bus stops with the maximum and minimum forecasted passenger count
st.write(f"Bus stop with the maximum forecasted passenger count: **{max_passenger_count_stop}**")
st.write(f"Maximum forecasted passenger count: **{int(max_passenger_count)}**")
st.write(f"Bus stop with the minimum forecasted passenger count: **{min_passenger_count_stop}**")
st.write(f"Minimum forecasted passenger count: **{int(min_passenger_count)}**")



