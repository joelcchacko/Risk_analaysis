import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch ESG scores via yfinance
def fetch_esg_scores(tickers):
    esg_scores = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            esg_scores[ticker] = stock.info.get('esgScores', {}).get('totalEsg', None)
        except Exception as e:
            print(f"Error fetching ESG score for {ticker}: {e}")
            esg_scores[ticker] = None
    return esg_scores

# Load portfolio data
portfolio = pd.read_csv('portfolio.csv')  # Portfolio file with Ticker, Weight

# Fetch ESG scores dynamically
tickers = portfolio['Ticker'].tolist()
esg_scores = fetch_esg_scores(tickers)

# Add ESG scores to the portfolio
portfolio['ESG_Score'] = portfolio['Ticker'].map(esg_scores)
portfolio['ESG_Score'].fillna(5, inplace=True)  # Default score for missing data

# Fetch historical price data
price_data = yf.download(tickers, start="2020-01-01", end="2023-12-31")['Adj Close']

# Calculate daily returns
returns = price_data.pct_change().dropna()

# Convert weights to a numpy array
portfolio_weights = portfolio.set_index('Ticker')['Weight'].values

# Calculate portfolio returns based on initial weights
portfolio_returns = (returns * portfolio_weights).sum(axis=1)

# Risk Metrics
risk_free_rate = 0.02  # Annualized risk-free rate
annualized_return = np.mean(portfolio_returns) * 252  # 252 trading days in a year
annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

# Portfolio Optimization: Simulate random weights to maximize Sharpe ratio
np.random.seed(42)  # For reproducibility

# Number of simulations
num_simulations = 10000
results = np.zeros((4, num_simulations))  # [Annualized Return, Volatility, Sharpe Ratio, Portfolio Weights]
max_sharpe_idx = 0
max_sharpe_ratio = -np.inf

# Simulate portfolios
for i in range(num_simulations):
    # Generate random portfolio weights (sum to 1)
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    
    # Calculate portfolio returns
    weighted_returns = returns.dot(weights)
    portfolio_annual_return = np.mean(weighted_returns) * 252
    portfolio_volatility = np.std(weighted_returns) * np.sqrt(252)
    
    # Calculate Sharpe ratio
    sharpe = (portfolio_annual_return - risk_free_rate) / portfolio_volatility
    
    # Store results
    results[0, i] = portfolio_annual_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe
    results[3, i] = i  # Track index of portfolio
    
    # Track maximum Sharpe ratio
    if sharpe > max_sharpe_ratio:
        max_sharpe_ratio = sharpe
        max_sharpe_idx = i

# Extract the optimal portfolio weights (with the highest Sharpe ratio)
optimal_weights = results[:, max_sharpe_idx]
optimal_portfolio_return = optimal_weights[0]
optimal_portfolio_volatility = optimal_weights[1]
optimal_sharpe_ratio = optimal_weights[2]

# Display the optimal portfolio weights
optimal_portfolio_weights = np.random.random(len(tickers))
optimal_portfolio_weights /= np.sum(optimal_portfolio_weights)  # Normalize to 1
optimal_weighted_tickers = dict(zip(tickers, optimal_portfolio_weights))


# Fetch ESG scores for the optimal portfolio
optimal_esg_scores = fetch_esg_scores(tickers)
portfolio['Optimal_Weight'] = optimal_portfolio_weights
portfolio['Optimal_ESG_Score'] = portfolio['Ticker'].map(optimal_esg_scores)
portfolio['Optimal_ESG_Score'].fillna(5, inplace=True)  # Fill missing ESG scores

# Scatter Plot: ESG Scores vs. Annualized Returns for Individual Stocks
plt.figure(figsize=(10, 6))
for ticker in portfolio['Ticker']:
    # Calculate annualized returns for each stock
    stock_annual_return = returns[ticker].mean() * 252
    stock_esg_score = portfolio.loc[portfolio['Ticker'] == ticker, 'ESG_Score'].values[0]
    
    # Plot ESG score vs. annualized return
    plt.scatter(stock_esg_score, stock_annual_return, label=ticker)


# Final Portfolio Metrics
print("\n=== Portfolio Risk Analysis ===")
print(f"Portfolio Annualized Return: {annualized_return:.2%}")
print(f"Portfolio Annualized Volatility: {annualized_volatility:.2%}")
print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Portfolio VaR (95% Confidence): {np.percentile(portfolio_returns, (1 - 0.95) * 100):.2%}")
print(f"Average Portfolio ESG Score: {np.average(portfolio['ESG_Score'], weights=portfolio['Weight']):.2f}")


print("\n=== Portfolio Optimization Results ===")
print(f"Optimal Portfolio Weights: {optimal_weighted_tickers}")
print(f"Maximum Sharpe Ratio: {optimal_sharpe_ratio:.2f}")
print(f"Portfolio Annualized Return: {optimal_portfolio_return:.2%}")
print(f"Portfolio Annualized Volatility: {optimal_portfolio_volatility:.2%}")

# Highlight the optimal portfolio
plt.axhline(optimal_portfolio_return, color='r', linestyle='--', label='Optimal Portfolio Return')
plt.axvline(np.mean(portfolio['Optimal_ESG_Score']), color='g', linestyle='--', label='Optimal Portfolio ESG Avg')
plt.title('ESG Scores vs. Annualized Returns (with Optimal Portfolio)')
plt.xlabel('ESG Score')
plt.ylabel('Annualized Return')
plt.legend()
plt.show()

