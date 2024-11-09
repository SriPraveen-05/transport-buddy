import pandas as pd
import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# Load the dataset
data = pd.read_csv("D://hackhustle//optimized_dispatch_final.csv")

# Parameters from dataset
routes = data['route_no'].tolist()
required_buses = data.set_index('route_no')['required_buses'].to_dict()

# Streamlit App
st.title('Bus Dispatch Optimization')

# Dropdown to select bus route
selected_route = st.selectbox('Select a Bus Route', routes)

# Display the required buses for the selected route
st.write(f"Route {selected_route} requires {required_buses[selected_route]} buses.")



