import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Reliance LSTM Prediction")

st.title("Reliance Stock Prediction using LSTM")

df = pd.read_csv("RELIANCE_2020-2025.csv")
# Clean column names
df.columns = df.columns.str.strip()

# Auto-detect Close column
close_col = None
for c in df.columns:
    if 'close' in c.lower() and 'prev' not in c.lower():
        close_col = c
        break
if close_col is None:
    close_col = 'Close Price'

# Convert Date
date_col = [c for c in df.columns if 'date' in c.lower()][0]
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.sort_values(date_col)

# CRITICAL FIX: Keep only EQ series and real prices
if 'Series' in df.columns:
    df = df[df['Series'].isin(['EQ','BE'])]
if df[close_col].astype(str).str.replace('.','',1).str.isdigit().all() == False:
    df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

df = df[df[close_col] > 50] # Remove 0.21, 0.23 junk
df = df.dropna(subset=[close_col, date_col])
df = df.drop_duplicates(subset=[date_col])

st.success(f"Loaded: {len(df)} rows from {df[date_col].min().date()} to {df[date_col].max().date()}")
st.dataframe(df.head())

close_prices = df[[close_col]].values.astype(float)

# LSTM
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

# Predict
last_60 = scaled[-60:]
future = []
curr = last_60.reshape(1,60,1)
for _ in range(30):
    pred = model.predict(curr, verbose=0)
    future.append(pred[0,0])
    curr = np.append(curr[:,1:,:], [[[pred[0,0]]]], axis=1)

future_prices = scaler.inverse_transform(np.array(future).reshape(-1,1))

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(y=close_prices.flatten(), name="Actual"))
fig.add_trace(go.Scatter(y=list(range(len(close_prices), len(close_prices)+30)),
                         x=list(range(len(close_prices), len(close_prices)+30)),
                         mode='lines', name="Future"))
# Better future plot
future_x = list(range(len(close_prices), len(close_prices)+30))
fig.add_trace(go.Scatter(x=future_x, y=future_prices.flatten(), name="Predicted Next 30 Days"))
st.plotly_chart(fig)
st.write("Next Day Prediction:", float(future_prices[0]))
