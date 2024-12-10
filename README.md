# Risk_analaysis
Portfolio Forecasting and Risk Assessment with Time Series and Monte Carlo Simulation

Overview

This Python scripts demonstrates how to forecast the future performance of a stock portfolio, evaluate its risk, and simulate potential future outcomes using time series forecasting (Prophet) and Monte Carlo simulations. It combines financial forecasting with a risk assessment model to provide insights into potential portfolio returns and their volatility. The script uses historical price data, ESG (Environmental, Social, and Governance) scores, and weights for each stock in the portfolio to conduct the analysis.

Libraries Used:
	•	yfinance: For fetching financial data (stock prices and ESG scores).
	•	numpy: For numerical operations, particularly in Monte Carlo simulations.
	•	pandas: For data manipulation and handling financial data.
	•	matplotlib: For plotting the results of forecasts and simulations.
	•	prophet: For time series forecasting using Facebook Prophet.

Features
	•	ESG Score Fetching: The script fetches ESG scores for stocks in the portfolio.
	•	Historical Data Download: Stock price data is fetched for each ticker in the portfolio.
	•	Time Series Forecasting (Prophet): Forecast future stock prices and returns based on historical price data using the Prophet model.
	•	Monte Carlo Simulation: Simulates multiple potential future paths for portfolio returns, taking into account forecasted returns and historical volatility.
	•	Risk Metrics Calculation: Calculates risk metrics, including the 5th and 95th percentiles of cumulative returns, annualized return, annualized volatility, and the Sharpe ratio.

Prerequisites

Before running the script, ensure you have the required libraries installed:

pip install yfinance numpy pandas matplotlib prophet

Input Data
	1.	Portfolio Data (portfolio.csv):
	•	A CSV file containing the portfolio details, with the following columns:
	•	Ticker: The stock ticker (e.g., AAPL, MSFT).
	•	Weight: The portfolio weight for each stock (e.g., 0.2 for 20%).
Example (portfolio.csv):

Ticker	Weight
AAPL	0.3
MSFT	0.4
AMZN	0.3


	2.	Stock Price Data: The script automatically fetches stock price data using yfinance for all tickers listed in the portfolio CSV file.

How it Works
	1.	Fetching ESG Scores:
The function fetch_esg_scores retrieves the total ESG score for each stock ticker in the portfolio using the yfinance API.
	2.	Time Series Forecasting:
The script uses Facebook’s Prophet library to forecast the future prices of each stock based on historical data. The forecasts are extended for a year (252 trading days).
	3.	Monte Carlo Simulation:
Monte Carlo simulations are performed to estimate the potential future portfolio returns. The script generates 10,000 simulation paths for each stock in the portfolio by using a normal distribution of returns based on the forecasted returns and historical volatility.
	4.	Risk Metrics:
After the Monte Carlo simulations, risk metrics are calculated:
	•	Mean Cumulative Return: The average return from the simulations.
	•	5th Percentile: The worst-case return scenario.
	•	95th Percentile: The best-case return scenario.
	•	Annualized Return: The expected return over a year.
	•	Annualized Volatility: The expected volatility of the portfolio over a year.
	•	Sharpe Ratio: A measure of portfolio performance relative to its risk.

Output
	•	Forecasted Stock Prices: A plot of the predicted stock prices for all tickers in the portfolio.
	•	Monte Carlo Simulations: A plot showing 10,000 possible future outcomes for the portfolio’s cumulative return.
	•	Risk Assessment Metrics: The mean, 5th percentile, and 95th percentile of cumulative returns, as well as the Sharpe ratio and annualized volatility.

Example Usage
	1.	Ensure your portfolio data (portfolio.csv) is correctly formatted and located in the same directory as the script.
	2.	Run the script to forecast prices, simulate portfolio returns, and calculate risk metrics.

python portfolio_forecasting.py

Expected Outputs

Forecasted Stock Prices Plot

A plot showing the forecasted prices for each ticker based on the Prophet model.

Monte Carlo Simulation Plot

A plot displaying 10,000 simulated paths for the portfolio’s cumulative return, allowing for an assessment of the possible future performance.

Risk Assessment Metrics:

=== Risk Assessment Metrics ===
Mean Cumulative Return after 1 year: 6.50%
5th Percentile (Worst-Case): -3.20%
95th Percentile (Best-Case): 15.80%

Portfolio Summary Metrics:

=== Portfolio Summary Metrics ===
Portfolio Annualized Return: 7.80%
Portfolio Annualized Volatility: 12.30%
Portfolio Sharpe Ratio: 0.62

Customization
	•	Forecast Horizon: You can adjust the forecast period (future_days variable) for more or fewer days.
	•	Number of Simulations: The number of Monte Carlo simulations (n_simulations) can be increased for more accurate results, though it will require more computational resources.
	•	Risk-Free Rate: Modify the risk_free_rate to reflect the current risk-free rate for the Sharpe ratio calculation.

License

This script is open-source and available for use and modification. Feel free to contribute or modify it for your specific needs.
