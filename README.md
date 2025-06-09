# 📈 Stock Market Analysis & Prediction with LSTM

This project uses Exploratory Data Analysis (EDA), Machine Learning (Random Forest), and Deep Learning (LSTM) to analyze and forecast stock prices for top companies like AAPL, MSFT, GOOG, and NFLX.

The project also includes a fully functional **Streamlit web app** to interactively predict stock prices.

---

## 🗂️ Project Structure
stock-market-analysis/
├── app/ # Streamlit app
├── data/ # CSV stock data
├── notebooks/ # Jupyter notebook with EDA & modeling
├── outputs/ # Trained LSTM model (.keras)
├── .gitignore # Files to exclude from Git
└── README.md # This file


---

## 🚀 Features

- 📊 **EDA**: Trends, correlation heatmaps, volatility
- 📉 **Random Forest Model**: Traditional regression with high R²
- 🔮 **LSTM Model**: Predicts next-day closing prices
- 🌐 **Streamlit App**:
  - Auto-loads latest 60 closing prices from CSV
  - Dropdown to select company (AAPL, MSFT, etc.)
  - Interactive prediction + line chart
  - CSV download of result

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/yourusername/stock-market-analysis.git
cd stock-market-analysis
pip install -r requirements.txt
streamlit run app/streamlit_app.py


