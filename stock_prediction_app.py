import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


def get_stock_data(ticker, start="2015-01-01", end="2024-01-01"):
    stock = yf.download(ticker, start=start, end=end)
    return stock


def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))
    
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    def create_sequences(data, seq_length=60):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    return X_train, y_train, X_test, y_test, scaler


def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60,1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(model, data, scaler):
    predictions = model.predict(data)
    return scaler.inverse_transform(predictions)

# Streamlit UI
st.title("📈 Stock Market Analysis & Prediction")

ticker = st.text_input("Enter Stock Ticker:", "")

if st.button("Fetch & Predict"):
    stock_data = get_stock_data(ticker)

    if stock_data is None or stock_data.empty:
        st.error("❌ Failed to fetch stock data. Please try another ticker.")
    else:
        st.write("✅ **Stock Data**", stock_data.tail())

        X_train, y_train, X_test, y_test, scaler = preprocess_data(stock_data)
        
        if X_train is None:
            st.error("⚠️ Not enough data for training. Try a different stock ticker.")
        else:
            model = build_lstm_model()
            with st.spinner("🔄 Training LSTM Model..."):
                model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

            predicted_prices = predict_future(model, X_test, scaler)

            # Plotting results
            plt.figure(figsize=(12,6))
            plt.plot(stock_data.index[-len(predicted_prices):], scaler.inverse_transform(y_test), label="Actual Price")
            plt.plot(stock_data.index[-len(predicted_prices):], predicted_prices, label="Predicted Price")
            plt.legend()
            plt.title(f"{ticker} Stock Price Prediction")
            st.pyplot(plt)
