import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Reliance LSTM")
st.title("LSTM - Reliance Stock Prediction")

df = pd.read_csv("RELIANCE_2020-2025.csv")
df.columns = df.columns.str.strip()

# Find cols
close_col = next((c for c in df.columns if 'close' in c.lower() and 'prev' not in c.lower()), df.columns[-4])
date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
df = df.dropna(subset=[close_col, date_col]).sort_values(date_col)

# Try to filter EQ if exists, else use all
if 'Series' in df.columns and (df['Series']=='EQ').any():
    df_eq = df[df['Series']=='EQ']
    if len(df_eq) > 60:
        df = df_eq

# Remove only extreme junk, not all
df = df[df[close_col] > 0]

if len(df) < 70:
    st.warning("CSV has too few rows. Downloading real Reliance data from Yahoo...")
    import yfinance as yf
    df = yf.download("RELIANCE.NS", period="5y", auto_adjust=True).reset_index()
    date_col = 'Date'
    close_col = 'Close'
    df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

st.success(f"Loaded: {len(df)} rows from {df[date_col].min()} to {df[date_col].max()}")
st.dataframe(df.tail())

close_prices = df[[close_col]].values.astype(float)
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

# Forecast
last = scaled[-60:].reshape(1,60,1)
preds = []
curr = last
for _ in range(30):
    p = model.predict(curr, verbose=0)[0,0]
    preds.append(p)
    curr = np.append(curr[:,1:,:], [[[p]]], axis=1)

future = scaler.inverse_transform(np.array(preds).reshape(-1,1))

fig = go.Figure()
fig.add_trace(go.Scatter(y=close_prices.flatten(), name="Actual"))
fig.add_trace(go.Scatter(x=list(range(len(close_prices), len(close_prices)+30)), y=future.flatten(), name="Predicted 30 Days"))
st.plotly_chart(fig)
st.metric("Next Day Prediction", f"Rs. {float(future[0][0]):.2f}")
