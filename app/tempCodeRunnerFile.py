import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

# ───────────────────────────────────────────────
# ✅ Page setup
st.set_page_config(page_title="AAPL Stock Prediction", layout="centered")
st.title("📈 Stock Price Prediction (LSTM)")
st.markdown("This app loads the latest 60 closing prices from `stocks.csv` and predicts the next day's price using an LSTM model.")

# ───────────────────────────────────────────────
# ✅ Select company
ticker = st.selectbox("Select a company:", ["AAPL", "MSFT", "GOOG", "NFLX"])

# ───────────────────────────────────────────────
# ✅ Load the LSTM model
model_path = r"H:\\stock_market_analysis\\outputs\\aapl_lstm_model.keras"
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}")
    st.stop()

model = load_model(model_path)

# ───────────────────────────────────────────────
# ✅ Load and process the stock data
csv_path = r"H:\\stock_market_analysis\\data\\stocks.csv"
try:
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter selected company and sort by date
    selected = df[df['Ticker'] == ticker].sort_values('Date')

    # Get the last 60 closing prices
    close_prices = selected['Close'].tail(60).values.reshape(-1, 1)

    if len(close_prices) < 60:
        st.error(f"❌ Not enough {ticker} data in CSV. At least 60 records required.")
        st.stop()

    # Scale and reshape
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)
    X_input = scaled.reshape(1, 60, 1)

    # Predict
    pred_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(pred_scaled)

    # Display result
    st.success(f"📉 Predicted Next {ticker} Closing Price: **${predicted_price[0][0]:.2f}**")

    # ───────────────────────────────────────────────
    # ✅ Line chart of last 60 prices
    st.subheader(f"📊 Last 60 {ticker} Closing Prices")
    st.line_chart(close_prices.flatten())

    # ───────────────────────────────────────────────
    # ✅ Export Prediction to CSV
    pred_df = pd.DataFrame({
        "Ticker": [ticker],
        "Predicted_Close": [round(predicted_price[0][0], 2)]
    })
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction as CSV", csv, f"{ticker}_predicted_price.csv", "text/csv")

except Exception as e:
    st.error(f"⚠️ Error while loading or processing data: {e}")

# ───────────────────────────────────────────────
# ✅ Deployment Tips
st.info("To deploy this app online, push your project to GitHub and connect to [Streamlit Cloud](https://streamlit.io/cloud)")
