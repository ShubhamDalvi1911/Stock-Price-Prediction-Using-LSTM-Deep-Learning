import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ---------------------------------------------------
# STEP 1: Download Stock Data
# ---------------------------------------------------

stock = "AAPL"   # Change stock symbol here

df = yf.download(stock, start="2015-01-01", end="2025-01-01")

print(df.head())

# Use only Close prices
data = df[['Close']]

# ---------------------------------------------------
# STEP 2: Scale Data
# ---------------------------------------------------

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(data)

# ---------------------------------------------------
# STEP 3: Create Dataset
# ---------------------------------------------------

X = []
y = []

prediction_days = 60

for i in range(prediction_days, len(scaled_data)):
    X.append(scaled_data[i-prediction_days:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------------------------------------------------
# STEP 4: Build LSTM Model
# ---------------------------------------------------

model = Sequential()

# First LSTM layer
model.add(LSTM(units=64, return_sequences=True,
               input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))

# Third LSTM layer
model.add(LSTM(units=64))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam',
              loss='mean_squared_error')

# ---------------------------------------------------
# STEP 5: Train Model
# ---------------------------------------------------

history = model.fit(
    X,
    y,
    epochs=25,
    batch_size=32
)

# ---------------------------------------------------
# STEP 6: Prepare Test Data
# ---------------------------------------------------

test_start = "2024-01-01"
test_end = "2025-01-01"

test_data = yf.download(stock, start=test_start, end=test_end)

actual_prices = test_data['Close'].values

# Combine datasets
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[
    len(total_dataset) - len(test_data) - prediction_days:
].values

model_inputs = model_inputs.reshape(-1, 1)

model_inputs = scaler.transform(model_inputs)

# Create test sequences
X_test = []

for i in range(prediction_days, len(model_inputs)):
    X_test.append(model_inputs[i-prediction_days:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(
    X_test,
    (X_test.shape[0], X_test.shape[1], 1)
)

# ---------------------------------------------------
# STEP 7: Predict Prices
# ---------------------------------------------------

predicted_prices = model.predict(X_test)

predicted_prices = scaler.inverse_transform(predicted_prices)

# ---------------------------------------------------
# STEP 8: Plot Results
# ---------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(actual_prices, color='black', label=f"Actual {stock} Price")

plt.plot(predicted_prices, color='green', label=f"Predicted {stock} Price")

plt.title(f"{stock} Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

plt.show()

# ---------------------------------------------------
# STEP 9: Predict Next Day Price
# ---------------------------------------------------

real_data = [model_inputs[len(model_inputs) - prediction_days:, 0]]

real_data = np.array(real_data)

real_data = np.reshape(
    real_data,
    (real_data.shape[0],
     real_data.shape[1],
     1)
)

prediction = model.predict(real_data)

prediction = scaler.inverse_transform(prediction)

print(f"\nNext Day Predicted Price: {prediction[0][0]:.2f}")
