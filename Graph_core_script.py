import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Fetch historical price data
tickers = portfolio['Ticker'].tolist()
price_data = yf.download(tickers, start="2015-01-01", end="2023-12-31")['Adj Close']

# Calculate daily returns
returns = price_data.pct_change().dropna()

# Convert weights to a numpy array
portfolio_weights = portfolio.set_index('Ticker')['Weight'].values

# Calculate portfolio returns based on initial weights
portfolio_returns = (returns * portfolio_weights).sum(axis=1)

# Individual annualized returns and volatility for tickers
individual_annual_returns = returns.mean() * 252
individual_volatility = returns.std() * np.sqrt(252)

# Final Portfolio Metrics
risk_free_rate = 0.02  # Annualized risk-free rate
annualized_return = np.mean(portfolio_returns) * 252  # 252 trading days in a year
annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

# Scatter Plot: Volatility vs. Annualized Returns
plt.figure(figsize=(10, 6))
plt.scatter(individual_volatility, individual_annual_returns, s=100, alpha=0.7, color='purple', edgecolor='k')
for i, ticker in enumerate(tickers):
    plt.text(individual_volatility[i], individual_annual_returns[i], ticker, fontsize=9, ha='right')
plt.title('Volatility vs. Annualized Returns')
plt.xlabel('Annualized Volatility (%)')
plt.ylabel('Annualized Return (%)')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# Bar Chart: Annualized Returns and Volatility
plt.figure(figsize=(12, 6))
bar_width = 0.35
x = np.arange(len(tickers))  # Ticker positions
plt.bar(x - bar_width/2, individual_annual_returns, width=bar_width, label='Annualized Return (%)', color='gold')
plt.bar(x + bar_width/2, individual_volatility, width=bar_width, label='Annualized Volatility (%)', color='skyblue')
plt.title('Annualized Returns and Volatility by Ticker')
plt.xlabel('Tickers')
plt.ylabel('Values (%)')
plt.xticks(x, tickers)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Pie Chart: Portfolio Weights
plt.figure(figsize=(8, 8))
weights = portfolio['Weight']
plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Portfolio Weights by Ticker')
plt.tight_layout()
plt.show()

# Line Chart: Cumulative Portfolio Returns
cumulative_returns = (1 + portfolio_returns).cumprod()
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Portfolio', color='green', linewidth=2)
plt.title('Cumulative Portfolio Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap: Correlation Matrix of Returns
correlation_matrix = returns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Returns')
plt.tight_layout()
plt.show()

# Final Summary Metrics
print("\n=== Portfolio Risk Analysis ===")
print(f"Portfolio Annualized Return: {annualized_return:.2%}")
print(f"Portfolio Annualized Volatility: {annualized_volatility:.2%}")
print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")
