import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

st.set_page_config(page_title="Reliance LSTM Predictor", layout="wide")
st.title("📈 RELIANCE - 6 Year LSTM Stock Predictor (2020-2025)")
st.caption("Dataset: 635 trading days | Model: 2-Layer LSTM")

# 1. LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("RELIANCE_2020-2025.csv")
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Find date and close columns automatically
    date_col = [c for c in df.columns if 'date' in c.lower()][0]
    close_col = [c for c in df.columns if 'close' in c.lower() or c.lower() == 'ltp'][0]

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df = df.sort_values(date_col).dropna(subset=[date_col])
    df = df.drop_duplicates(subset=[date_col])
    return df, date_col, close_col

try:
    df, DATE_COL, CLOSE_COL = load_data()
    st.success(f"Loaded: {len(df)} rows from {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
    st.dataframe(df.tail(), use_container_width=True)
except Exception as e:
    st.error(f"Upload RELIANCE_2020-2025.csv to your repo. Error: {e}")
    st.stop()

# 2. PREPARE DATA FOR LSTM
close_prices = df[[CLOSE_COL]].values.astype(float)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close_prices)

SEQ_LEN = 60
X, y = [], []
for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. BUILD MODEL
@st.cache_resource
def build_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model()

# 4. TRAIN & PREDICT
if st.button("🚀 Train LSTM Model", type="primary"):
    with st.spinner("Training on 6 years data... ~1 min"):
        history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                          validation_data=(X_test, y_test), verbose=0)

    st.success("Training Complete!")

    # Prediction for chart
    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Chart: Actual vs Predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual.flatten()[-100:], name="Actual", line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=pred.flatten()[-100:], name="Predicted (LSTM)", line=dict(color='red')))
    fig.update_layout(title="Last 100 Days: Actual vs LSTM Predicted", xaxis_title="Days", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

    # Tomorrow's Prediction
    last_60 = scaled[-60:].reshape(1, SEQ_LEN, 1)
    tomorrow_scaled = model.predict(last_60)
    tomorrow_price = scaler.inverse_transform(tomorrow_scaled)[0][0]
    last_price = close_prices[-1][0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Close", f"₹{last_price:.2f}")
    col2.metric("Predicted Next Close", f"₹{tomorrow_price:.2f}", f"{tomorrow_price-last_price:.2f}")
    col3.metric("Expected Change", f"{((tomorrow_price-last_price)/last_price)*100:.2f}%")

else:
    st.info("Click 'Train LSTM Model' to start prediction")