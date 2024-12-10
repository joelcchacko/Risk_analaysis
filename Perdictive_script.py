import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

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
price_data = yf.download(tickers, start="2015-01-01", end="2024-12-01")['Adj Close']

# Calculate daily returns
returns = price_data.pct_change().dropna()

# Convert weights to a numpy array
portfolio_weights = portfolio.set_index('Ticker')['Weight'].values

# Time Series Forecasting (Prophet)
forecasted_returns = {}
future_days = 252  # Predict for 1 year (trading days)

print("\n=== Time Series Forecasting: Understanding Trends and Seasonality ===")
for ticker in tickers:
    try:
        # Prepare data
        df = price_data[ticker].reset_index()
        df.columns = ['ds', 'y']  # Prophet requires 'ds' for dates and 'y' for values

        # Initialize and fit the model
        model = Prophet()
        model.fit(df)

        # Forecast future prices
        future = model.make_future_dataframe(periods=future_days)
        forecast = model.predict(future)

        # Calculate forecasted daily returns
        forecast['daily_return'] = forecast['yhat'].pct_change()
        forecasted_returns[ticker] = forecast['daily_return'].iloc[-future_days:].values

        # Plot the forecast
        model.plot(forecast)
        plt.title(f'Forecasted Prices for {ticker}')
        plt.show()

    except Exception as e:
        print(f"Error forecasting for {ticker}: {e}")

# Monte Carlo Simulation
print("\n=== Monte Carlo Simulation: Assessing Risks and Probable Scenarios ===")
n_simulations = 10000  # Number of simulation paths
simulation_days = future_days  # Simulate for 1 year

# Simulated portfolio returns
simulated_portfolio_returns = []

for i in range(n_simulations):
    simulated_daily_returns = []

    for ticker in tickers:
        if ticker in forecasted_returns:
            # Get forecasted returns and historical volatility
            base_returns = forecasted_returns[ticker]
            historical_volatility = returns[ticker].std()

            # Generate random daily returns using normal distribution
            random_returns = np.random.normal(
                loc=base_returns.mean(),
                scale=historical_volatility,
                size=simulation_days
            )
            simulated_daily_returns.append(random_returns)

    # Calculate portfolio return for each simulation path
    portfolio_simulation = (np.array(simulated_daily_returns).T * portfolio_weights).sum(axis=1)
    simulated_portfolio_returns.append((1 + portfolio_simulation).cumprod())

# Convert simulation results to a DataFrame
simulated_portfolio_df = pd.DataFrame(simulated_portfolio_returns).T

# Plot Monte Carlo Simulations
plt.figure(figsize=(12, 6))
plt.plot(simulated_portfolio_df, alpha=0.1, color='blue')
plt.title('Monte Carlo Simulations of Portfolio Returns with Forecasting')
plt.xlabel('Days')
plt.ylabel('Cumulative Return')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# Analyze Risk Metrics
final_cumulative_returns = simulated_portfolio_df.iloc[-1]  # Final values after simulation
mean_cumulative_return = final_cumulative_returns.mean()
percentile_5 = final_cumulative_returns.quantile(0.05)
percentile_95 = final_cumulative_returns.quantile(0.95)

# Print Risk Assessment Metrics
print("\n=== Risk Assessment Metrics ===")
print(f"Mean Cumulative Return after 1 year: {mean_cumulative_return:.2%}")
print(f"5th Percentile (Worst-Case): {percentile_5:.2%}")
print(f"95th Percentile (Best-Case): {percentile_95:.2%}")

# Summary of Portfolio
risk_free_rate = 0.02  # Annualized risk-free rate
annualized_return = mean_cumulative_return  # Mean return from simulations
annualized_volatility = simulated_portfolio_df.pct_change().std().mean() * np.sqrt(252)
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

print("\n=== Portfolio Summary Metrics ===")
print(f"Portfolio Annualized Return: {annualized_return:.2%}")
print(f"Portfolio Annualized Volatility: {annualized_volatility:.2%}")
print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")
