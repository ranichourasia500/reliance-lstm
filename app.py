import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

st.set_page_config(page_title="Reliance LSTM")
st.title("LSTM - Reliance Stock Prediction (2020-2025)")

st.info("Downloading real Reliance data...")

df = yf.download("RELIANCE.NS", period="5y", auto_adjust=True).reset_index()
df = df.dropna()

st.success(f"Loaded: {len(df)} real trading days from {df['Date'].min().date()} to {df['Date'].max().date()}")
st.dataframe(df.tail())

close_prices = df[['Close']].values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close_prices)

X,y = [],[]
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i])
    y.append(scaled[i])
X,y = np.array(X), np.array(y)

model = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)), LSTM(50), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X,y, epochs=5, batch_size=32, verbose=0)

last = scaled[-60:].reshape(1,60,1)
preds = []
curr = last
for _ in range(30):
    p = model.predict(curr, verbose=0)[0,0]
    preds.append(p)
    curr = np.append(curr[:,1:,:], [[[p]]], axis=1)

future = scaler.inverse_transform(np.array(preds).reshape(-1,1))

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=close_prices.flatten(), name="Actual Price (Rs)"))
future_dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]
fig.add_trace(go.Scatter(x=future_dates, y=future.flatten(), name="Predicted Next 30 Days", line=dict(color='orange', dash='dash')))
fig.update_layout(title="Reliance (NSE) - 5 Year Actual vs 30 Day LSTM Forecast", xaxis_title="Date", yaxis_title="Price (Rs)")
st.plotly_chart(fig, use_container_width=True)

st.metric("Next Day Prediction", f"Rs. {float(future[0][0]):.2f}")
st.metric("30-Day Forecast (Last)", f"Rs. {float(future[-1][0]):.2f}")
