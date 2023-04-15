from sklearn.linear_model import LinearRegression

import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

yf.pdr_override()
company = "YNDX"
df = pdr.get_data_yahoo(company, start="2019-01-01", end="2022-04-15")

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train_data = data[0:train_size, :]
test_data = data[train_size:len(data), :]


def create_dataset(dataset, time_step=1):
    X_data, y_data = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X_data.append(a)
        y_data.append(dataset[i + time_step, 0])
    return np.array(X_data), np.array(y_data)


time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(test_predict, color='red', label='Predicted Stock Price')
plt.legend()
plt.show()


def predict_stock_price(company, stock_data):
    # Getting the latest stock data
    X_train = stock_data[['Open', 'High', 'Low']]
    y_train = stock_data['Close']

    # Creating and training the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Getting the latest stock data
    stock = yf.Ticker(company)
    latest_data = stock.history(period="1d")
    latest_data = latest_data.reset_index()
    latest_data = latest_data[['Open', 'High', 'Low']]

    # Predicting the closing price of the stock for tomorrow
    predicted_price = model.predict(latest_data)

    return predicted_price

    return predicted_price


# Example usage
predicted_price = predict_stock_price(company, df)
print('Predicted stock price for tomorrow: $', predicted_price[0])
