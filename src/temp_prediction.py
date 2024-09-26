import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Load the data
try:
    data = pd.read_csv('data/GlobalTemperatures.csv', skiprows=4)  # Skip the first 4 lines of metadata
    print(data)
except FileNotFoundError:
    print("The file 'GlobalTemperatures.csv' was not found. Please ensure it is in the 'data' directory.")
    exit()
except pd.errors.ParserError as e:
    print(f"Error parsing the CSV file: {e}")
    exit()

# Verify the data
print(data.head())

# Parse dates and set the index
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m')
data.set_index('Date', inplace=True)

# Select the temperature column (e.g., 'Value')
temperature_data = data['Value'].dropna()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(temperature_data, label='Temperature')
plt.title('Contiguous U.S. May Average Temperature Over Time')
plt.xlabel('Year')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.show()

# Split the data into training and testing sets
train_size = int(len(temperature_data) * 0.8)
train, test = temperature_data[:train_size], temperature_data[train_size:]

# Fit the ARIMA model
model = ARIMA(temperature_data, order=(1, 1, 1))
model_fit = model.fit()

# Make predictions
#predictions = model_fit.forecast(steps=len(test))
predictions = model_fit.predict(start=115, end= 125,dynamic=True)
predictions = pd.Series(predictions, index=test.index)
print(predictions)

# Plot the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions')
plt.title('ARIMA Model Predictions')
plt.xlabel('Year')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.show()

# Print the model summary
#print(model_fit.summary())
