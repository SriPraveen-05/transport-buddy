import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import daal4py as d4p
import numpy as np
import warnings
import joblib

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r"C:\Users\jegaa\OneDrive\Documents\uk_gov_data_sparse_preproc.csv", encoding='ISO-8859-1')
X = df[['engine_size_cm3', 'power_ps', 'fuel', 'transmission_type']]
y = df['co2_emissions_gPERkm']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numeric and categorical features
numeric_features = ['engine_size_cm3', 'power_ps']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['fuel', 'transmission_type']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))
])

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Preprocess the training and test data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Train a Scikit-learn Linear Regression model
from sklearn.linear_model import LinearRegression
sklearn_model = LinearRegression()
sklearn_model.fit(X_train_preprocessed, y_train)

# Save the model and preprocessor to .pkl files
joblib.dump(sklearn_model, 'linear_regression_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

# Evaluate the Scikit-learn model
y_pred_sklearn = sklearn_model.predict(X_test_preprocessed)
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)
print('Scikit-learn Model - Mean Absolute Error (MAE):', mae_sklearn)
print('Scikit-learn Model - R-squared:', r2_sklearn)
    
# Function to predict emissions using the saved model and preprocessor
def predict_emissions_sklearn():
    # Load the saved model and preprocessor
    model = joblib.load('linear_regression_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    
    # Get user input
    engine_size = float(input("Enter the engine size in cm³ : "))
    power_ps = float(input("Enter the power in PS : "))
    fuel = input("Enter the fuel type : ").strip().capitalize()
    transmission = input("Enter the transmission type : ").strip().capitalize()
    
    # Prepare input data
    new_car = pd.DataFrame({
        'engine_size_cm3': [engine_size],
        'power_ps': [power_ps],
        'fuel': [fuel],
        'transmission_type': [transmission]
    })
    
    # Preprocess input data
    new_car_preprocessed = preprocessor.transform(new_car)
    
    # Make prediction
    predicted_co2 = model.predict(new_car_preprocessed)
    print(f"Predicted CO2 Emissions for the new car: {predicted_co2[0]:.2f} grams per kilometer")

# Call the function to predict
predict_emissions_sklearn()
