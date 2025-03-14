import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Load stock list from Excel
@st.cache_data
def load_stocklist(file_path):
    df_dict = pd.read_excel(file_path, sheet_name=None)
    sheet_names = list(df_dict.keys()) 
    return df_dict, sheet_names

# Load macroeconomic data
@st.cache_data
def load_macro_data():
    macro_data = {
        "Date": pd.date_range(start="2023-01-01", periods=27, freq="M"),
        "GDP": [6.3, 6.5, 6.1, 5.8, 5.9, 6, 5.8, 6.1, 5.7, 5.5, 5.3, 5.6, 6.2, 6.3, 6, 6.1, 5.9, 5.7, 6, 6.2, 6.4, 6.6, 6.8, 7, 7.2, 7.4, 7.6],
        "Inflation": [6.16, 6.16, 5.79, 5.09, 4.42, 5.57, 7.54, 6.91, 5.02, 4.87, 5.55, 5.69, 5.10, 5.09, 4.85, 4.83, 4.75, 5.08, 3.54, 3.65, 5.49, 5.50, 3.60, 4.20, 4.00, 5.00, 4.50],
        "Interest Rate": [6.5] * 27  # Constant 6.5%
    }
    return pd.DataFrame(macro_data)

# Fetch stock data
def fetch_stock_data(tickers, period="1y"):
    data = yf.download(tickers, period=period)['Close']
    return data

# Arbitrage Pricing Theory (APT) Model with User Inputs
def calculate_apt(tickers, gdp, inflation, interest_rate):
    risk_free_rate = 0.04  # 4% Risk-Free Rate

    results = []
    for ticker in tickers:
        beta_inflation = np.random.uniform(0.5, 1.5)  # Simulated Beta for Inflation
        beta_gdp = np.random.uniform(0.5, 1.5)  # Simulated Beta for GDP Growth
        beta_interest = np.random.uniform(0.5, 1.5)  # Simulated Beta for Interest Rates
        
        expected_return = (risk_free_rate +
                           beta_inflation * (inflation / 100) +
                           beta_gdp * (gdp / 100) +
                           beta_interest * (interest_rate / 100))
        
        results.append({
            "Stock": ticker,
            "Beta Inflation": round(beta_inflation, 2),
            "Beta GDP": round(beta_gdp, 2),
            "Beta Interest": round(beta_interest, 2),
            "Expected Return (%)": round(expected_return * 100, 2)
        })

    return pd.DataFrame(results)

# Streamlit UI
st.title("📈 APT Model with User-Defined Macroeconomic Inputs")

# Load stock data
file_path = "stocklist.xlsx"
stock_data_dict, sheet_names = load_stocklist(file_path)

# User selects the sheet
sheet_selected = st.selectbox("Select Stock List Sheet:", sheet_names)
stock_df = stock_data_dict[sheet_selected]  
tickers = stock_df["Symbol"].dropna().tolist()

# User inputs macroeconomic assumptions
st.subheader("📊 Adjust Macroeconomic Inputs")
gdp_input = st.number_input("Expected GDP Growth (%)", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
inflation_input = st.number_input("Expected Inflation Rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
interest_input = st.number_input("Expected Interest Rate (%)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)

# Run APT model
if st.button("Run APT Model"):
    apt_df = calculate_apt(tickers, gdp_input, inflation_input, interest_input)
    st.subheader("📊 APT Expected Returns")
    st.dataframe(apt_df)
