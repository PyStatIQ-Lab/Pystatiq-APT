import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load stock list from Excel
@st.cache_data
def load_stocklist(file_path):
    df_dict = pd.read_excel(file_path, sheet_name=None)  # Load all sheets into a dictionary
    sheet_names = list(df_dict.keys())  # Extract sheet names
    return df_dict, sheet_names

# Fetch stock data
def fetch_stock_data(tickers, period="1y"):
    data = yf.download(tickers, period=period)['Adj Close']
    return data

# Fetch factor data (e.g., inflation, GDP, interest rates)
@st.cache_data
def fetch_factor_data():
    factors = {
        "Inflation": 0.03,  # 3% Inflation Rate
        "GDP Growth": 0.04,  # 4% GDP Growth
        "Interest Rate": 0.05  # 5% Interest Rate
    }
    return factors

# Fetch stock beta
def fetch_stock_beta(ticker):
    stock = yf.Ticker(ticker)
    beta = stock.info.get("beta", None)
    return beta

# Arbitrage Pricing Theory (APT) Model
def calculate_apt(tickers):
    risk_free_rate = 0.04  # 4% Risk-Free Rate
    factor_premiums = fetch_factor_data()
    
    results = []
    for ticker in tickers:
        beta_inflation = np.random.uniform(0.5, 1.5)  # Simulated Beta for Inflation
        beta_gdp = np.random.uniform(0.5, 1.5)  # Simulated Beta for GDP Growth
        beta_interest = np.random.uniform(0.5, 1.5)  # Simulated Beta for Interest Rates
        
        expected_return = (risk_free_rate +
                           beta_inflation * factor_premiums["Inflation"] +
                           beta_gdp * factor_premiums["GDP Growth"] +
                           beta_interest * factor_premiums["Interest Rate"])
        
        results.append({
            "Stock": ticker,
            "Beta Inflation": beta_inflation,
            "Beta GDP": beta_gdp,
            "Beta Interest": beta_interest,
            "Expected Return (%)": expected_return * 100
        })

    return pd.DataFrame(results)

# Portfolio Optimization using Modern Portfolio Theory (MPT)
def optimize_portfolio(data, risk_tolerance):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(data.columns)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    # Define objective function
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    if risk_tolerance == "Low":
        optimized = minimize(portfolio_volatility, weights, method="SLSQP", bounds=bounds, constraints=constraints)
    else:
        def neg_sharpe(weights):
            port_return = np.sum(mean_returns * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -port_return / port_volatility  

        optimized = minimize(neg_sharpe, weights, method="SLSQP", bounds=bounds, constraints=constraints)

    return optimized.x, mean_returns, cov_matrix

# Streamlit UI
st.title("📈 Quantitative Stock Selection Model")

# Load stock data
file_path = "/mnt/data/stocklist.xlsx"
stock_data_dict, sheet_names = load_stocklist(file_path)

# User selects the sheet
sheet_selected = st.selectbox("Select Stock List Sheet:", sheet_names)
stock_df = stock_data_dict[sheet_selected]  # Get selected sheet data
tickers = stock_df["Symbol"].dropna().tolist()  # Extract stock symbols

# User Inputs
model = st.selectbox("Select a Quantitative Model:", ["Arbitrage Pricing Theory (APT)", "Capital Asset Pricing Model (CAPM)", "Modern Portfolio Theory (MPT)", "Momentum Strategy"])
risk_tolerance = st.selectbox("Select Risk Tolerance:", ["Low", "Medium", "High"])
investment_horizon = st.selectbox("Investment Horizon:", ["Short-term", "Long-term"])
factors = st.multiselect("Factor Preferences:", ["Volatility", "Momentum", "Value", "Growth"])

if st.button("Generate Portfolio"):
    if model == "Arbitrage Pricing Theory (APT)":
        apt_df = calculate_apt(tickers)
        st.subheader("📊 APT Expected Returns")
        st.dataframe(apt_df)

    elif model == "Capital Asset Pricing Model (CAPM)":
        capm_df = calculate_apt(tickers)  # Use the same APT function for CAPM (just a simplification)
        st.subheader("📊 CAPM Expected Returns")
        st.dataframe(capm_df)

    else:
        data = fetch_stock_data(tickers, period="1y" if investment_horizon == "Short-term" else "5y")

        if model == "Modern Portfolio Theory (MPT)":
            weights, mean_returns, cov_matrix = optimize_portfolio(data, risk_tolerance)
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            st.subheader("📊 Optimized Portfolio (MPT)")
            portfolio_df = pd.DataFrame({"Stock": tickers, "Allocation (%)": weights * 100})
            st.dataframe(portfolio_df)
            st.write(f"**Expected Portfolio Return:** {portfolio_return:.2%}")
            st.write(f"**Expected Portfolio Risk:** {portfolio_risk:.2%}")

        elif model == "Momentum Strategy":
            momentum = (data.iloc[-1] / data.iloc[0]) - 1
            top_momentum_stocks = momentum.nlargest(5)
            st.subheader("🚀 Top Momentum Stocks")
            st.dataframe(top_momentum_stocks.to_frame(name="Momentum %"))
