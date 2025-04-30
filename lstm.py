import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📊 Stock Price Prediction using LSTM")
st.markdown("Upload your stock price dataset with `Date` and `Close` columns.")

# File upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Parse date and sort
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

    if 'Close' not in df.columns:
        st.error("The file must contain a 'Close' column.")
        st.stop()

    df.dropna(subset=['Close'], inplace=True)

    st.subheader("📄 Uploaded Data Preview")
    st.write(df.tail())

    # Plotting
    def plot_history():
        fig, ax = plt.subplots()
        ax.plot(df['Date'], df['Close'], label='Closing Price')
        ax.set_title("Stock Closing Price History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

    plot_history()

    # Data preprocessing
    data = df[['Close']]
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - 60:]

    def create_sequences(data, window=60):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i - window:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("⏳ Training the LSTM model..."):
        model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Prediction plot
    def plot_predictions():
        fig2, ax2 = plt.subplots()
        ax2.plot(actual, label='Actual')
        ax2.plot(predictions, label='Predicted')
        ax2.set_title("Predicted vs Actual Closing Prices")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        st.pyplot(fig2)

    st.subheader("📉 Model Predictions")
    plot_predictions()

    # Next-day prediction
    last_60 = scaled_data[-60:]
    X_future = last_60.reshape(1, 60, 1)
    future_pred = model.predict(X_future)
    future_price = scaler.inverse_transform(future_pred)

    st.subheader("📅 Next Day Price Prediction")
    st.success(f"Predicted Closing Price: ₹{future_price[0][0]:.2f}")

else:
    st.info("Please upload a CSV file to begin.")
