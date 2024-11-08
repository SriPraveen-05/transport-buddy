import pandas as pd
import numpy as np
import streamlit as st
from pmdarima import auto_arima
import matplotlib.pyplot as plt

# Load the dataset containing bus stops and hourly passenger counts
data = pd.read_csv("E://hackhustle2024//bus_stops_with_passenger_counts.csv")
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# Streamlit: User input for date range
st.title('Passenger Count Prediction for Bus Stops')
start_date = st.date_input("Start Date", value=data['date'].min())
end_date = st.date_input("End Date", value=data['date'].max())

# Filter the data for the selected date range
filtered_data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]

# Set up empty dictionaries to store the forecasts for each bus stop
bus_stop_forecasts = {}

# Loop over each bus stop in the filtered data
for bus_stop in filtered_data['bus_stop'].unique()[:15]:
    bus_stop_data = filtered_data[filtered_data['bus_stop'] == bus_stop]

    # Sum passenger counts per day to get daily data
    daily_passenger_counts = bus_stop_data.groupby('date').sum().filter(regex='passenger_count_hour').sum(axis=1)

    # Handle missing values
    daily_passenger_counts = daily_passenger_counts.fillna(0)

    # Use AutoARIMA to find the best parameters
    model = auto_arima(daily_passenger_counts, seasonal=True, m=7, trace=True, error_action='ignore', suppress_warnings=True)

    # Forecast for the next 7 days
    forecast = model.predict(n_periods=7)
    forecast_dates = pd.date_range(start=end_date, periods=7, freq='D')

    # Store forecast results for each bus stop
    bus_stop_forecasts[bus_stop] = {
        "forecast_mean": forecast,
        "forecast_dates": forecast_dates
    }

# Find the bus stop with the maximum and minimum forecasted passenger count
max_passenger_count_stop = None
min_passenger_count_stop = None
max_passenger_count = -np.inf
min_passenger_count = np.inf

# Loop to find the maximum and minimum forecasts
for bus_stop, forecast_data in bus_stop_forecasts.items():
    forecast_mean = forecast_data["forecast_mean"]
    forecast_max = forecast_mean.max()
    forecast_min = forecast_mean.min()

    if forecast_max > max_passenger_count:
        max_passenger_count = forecast_max
        max_passenger_count_stop = bus_stop

    if forecast_min < min_passenger_count:
        min_passenger_count = forecast_min
        min_passenger_count_stop = bus_stop

# Display the bus stops with the maximum and minimum forecasted passenger count
st.write(f"Bus stop with the maximum forecasted passenger count: *{max_passenger_count_stop}*")
st.write(f"Maximum forecasted passenger count: *{max_passenger_count}*")
st.write(f"Bus stop with the minimum forecasted passenger count: *{min_passenger_count_stop}*")
st.write(f"Minimum forecasted passenger count: *{min_passenger_count}*")

# Plotting the forecast for maximum and minimum passenger count bus stops
plt.figure(figsize=(10, 6))

# Plot for the bus stop with maximum forecasted passenger count
max_forecast = bus_stop_forecasts[max_passenger_count_stop]
plt.subplot(2, 1, 1)
plt.plot(max_forecast["forecast_dates"], max_forecast["forecast_mean"], label=f"Forecasted Passenger Count - {max_passenger_count_stop}", color='blue')
plt.title(f"Forecasted Passenger Count for {max_passenger_count_stop} (Maximum)")
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.legend()

# Plot for the bus stop with minimum forecasted passenger count
min_forecast = bus_stop_forecasts[min_passenger_count_stop]
plt.subplot(2, 1, 2)
plt.plot(min_forecast["forecast_dates"], min_forecast["forecast_mean"], label=f"Forecasted Passenger Count - {min_passenger_count_stop}", color='red')
plt.title(f"Forecasted Passenger Count for {min_passenger_count_stop} (Minimum)")
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.legend()

plt.tight_layout()
st.pyplot(plt)