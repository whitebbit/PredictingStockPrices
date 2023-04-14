import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def get_data(company, start=dt(2012, 1, 1), end=dt.now()):
    yf.pdr_override()
    return pdr.get_data_yahoo(company, start=start, end=end)


def get_trains(data, prediction_days, scaler=None):
    current_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1)) if scaler else data
    x_train = np.array([current_data[x - prediction_days:x, 0] for x in range(prediction_days, len(current_data))])
    y_train = np.array([current_data[y, 0] for y in range(prediction_days, len(current_data))])
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train


def get_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    return model


def get_validate(company, data, model, scaler, prediction_days):
    test_data = get_data(company, dt(2020, 1, 1))
    actual_prices = test_data["Close"].values

    total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test, y_test = get_trains(model_inputs, prediction_days)

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plt.plot(actual_prices, color="black", label=f"Actual {company} prices")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} prices")
    plt.title(f"{company} Share Price")
    plt.xlabel("Time")
    plt.legend()

    plt.show()
    return model_inputs


def get_prediction(model_inputs, model, scaler):
    real_data = np.array([model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]])
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction


if __name__ == "__main__":
    company = "GS"
    prediction_days = 60
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = get_data(company, dt(2012, 1, 1), dt(2020, 1, 1))
    x, y = get_trains(data, prediction_days, scaler)
    model = get_model(x, y)
    model_inputs = get_validate(company, data, model, scaler, prediction_days)
    prediction = get_prediction(model_inputs, model, scaler)
    print(f"Prediction: {prediction}")

