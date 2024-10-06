import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class PortfolioOptimizer:

    def __init__(self, returns, matrix, rf=0.03):  # the default risk-free rate is 3%
        self.returns = returns
        self.matrix = matrix
        self.rf = rf

    def get_portfolio_return(self, weight):
        return weight @ self.returns

    def get_portfolio_vol(self, weight):
        return np.sqrt(weight @ self.matrix @ weight.T)

    def get_portfolio_negative_sharpe_ratio(self, weight):
        port_sr = (self.get_portfolio_return(weight) - self.rf) / self.get_portfolio_vol(weight)
        return -port_sr


# Define the date range and stock tickers
start_date = "2024-01-01"
end_date = "2024-06-30"
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
stocks.sort()

# Download stock data
data = yf.download(stocks, start=start_date, end=end_date)["Adj Close"]

# Calculate daily returns
daily_R = np.log(data) - np.log(data.shift(1))
daily_R = daily_R.dropna()

# Calculate annualized returns and covariance matrix
avg_R = daily_R.mean()
annual_R = avg_R * 252
M = daily_R.cov() * 252

# Initialize portfolio optimizer
MPT = PortfolioOptimizer(annual_R, M, rf=0.05)  # please set the current risk-free rate here, e.g. 5%

# Initial weights
no_of_stocks = len(stocks)
W = np.array([1 / no_of_stocks] * no_of_stocks)

# Optimization
bounds = tuple((0, 1) for i in range(no_of_stocks))
constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})

result = minimize(
    fun=MPT.get_portfolio_negative_sharpe_ratio,
    x0=W,
    bounds=bounds,
    constraints=constraints
)

ratios = result.x
opt_ratios = [round(ratio, 4) for ratio in ratios]

# Display optimized weights
for stock, ratio in zip(stocks, opt_ratios):
    print(f"{stock}: {ratio * 100:.2f}%")

# Display optimized Sharpe Ratio
opt_sr = -1 * result.fun
print(f"\nOptimized Sharpe Ratio: {opt_sr}")

# Monte Carlo simulation
no_of_samples = 1000
rand_return_list = []
rand_SD_list = []
rand_SR_list = []

for sample in range(no_of_samples):
    rand_weight = np.random.random(no_of_stocks)
    rand_weight /= np.sum(rand_weight)

    rand_return = MPT.get_portfolio_return(rand_weight)
    rand_SD = MPT.get_portfolio_vol(rand_weight)
    rand_SR = -1 * MPT.get_portfolio_negative_sharpe_ratio(rand_weight)  # reverse back

    rand_return_list.append(rand_return)
    rand_SD_list.append(rand_SD)
    rand_SR_list.append(rand_SR)

# Find optimal portfolio from simulation
index = np.argmax(rand_SR_list)
opt_weight = rand_SR_list[index]
print(opt_weight)

# Plotting
plt.figure(figsize=(16, 8))
plt.scatter(rand_SD_list, rand_return_list, c=rand_SR_list, cmap="viridis", marker="o")
plt.colorbar(label='Sharpe Ratio')
plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.scatter(rand_SD_list[index], rand_return_list[index], c="red", marker='*', s=200)
plt.show()
