import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from streamlit_tags import st_tags
from scipy import optimize

st.set_page_config(layout="wide")
st.title("Portfolio Balancer")
st.write(
    """
1. Designed for balancing of a portfolio of stocks by maximizing it's the [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp).  
2. Input your preferred portfolio with constraints and compute allocation with the highest Ratio.
3. Sharpe is just one ratio, consider exploring others such as [Sortino](https://www.investopedia.com/terms/s/sortinoratio.asp) or [Treynor](https://www.investopedia.com/terms/t/treynorratio.asp).
4. _None of this is financial advice, just a tool for DYOR._
"""
)
st.write(
    """
    Made with ❤️ by [Conrad McGee Stocks](https://www.linkedin.com/in/mcgeestocks/)
    """
)


@st.cache_data
def fetch_data(
    tickerlist=[],
    period="1y",
    interval="1d",
):
    """
    Fetches data from Yahoo Finance API
    """
    holder = []
    for ticker in tickerlist:
        holder.append(
            yf.Ticker(ticker).history(period=period, interval=interval, actions=False)[
                "Close"
            ]
        )
    return pd.DataFrame(zip(*holder), index=holder[0].index, columns=[*tickerlist])


def montecarlo(df):
    """
    Monte Carlo simulation for portfolio optimization
    """
    sims = 10000
    all_weights = np.zeros((sims, len(df.columns)))
    ret_arr = np.zeros(sims)
    vol_arr = np.zeros(sims)
    sharpe_arr = np.zeros(sims)
    log_returns = np.log(df / df.shift(1))

    for sim in range(sims):
        weights = np.array(np.random.random(len(df.columns)))
        weights = weights / np.sum(weights)
        all_weights[sim, :] = weights

        ret_arr[sim] = np.sum((log_returns.mean() * weights) * 251)
        vol_arr[sim] = np.sqrt(
            np.dot(weights.T, np.dot(log_returns.cov() * 251, weights))
        )
        sharpe_arr[sim] = ret_arr[sim] / vol_arr[sim]

    return sharpe_arr, ret_arr * 100, vol_arr * 100, all_weights


def maximize_sharpe(
    mean_returns,
    covar_returns,
    risk_free_rate,
    portfolio_size,
    lower_bound,
    upper_bound,
):
    # define maximization of Sharpe Ratio using principle of duality
    def objective_function(
        x, mean_returns, covar_returns, risk_free_rate, portfolio_size
    ):
        """
        Objective function for maximization of the sharpe ratio.
        for more info see: https://www.researchgate.net/publication/335517021_Python_for_Portfolio_Optimization_The_Ascent
        """
        numerator = np.matmul(np.array(mean_returns), x.T) - risk_free_rate
        denominator = np.sqrt(np.matmul(np.matmul(x, covar_returns), x.T))
        return -(numerator / denominator)

    def equality_constraint(x):
        """
        Equality constraint for portfolio weights
        """
        A = np.ones(x.shape)
        b = 1
        return np.matmul(A, x.T) - b

    initial_guess = np.repeat(1 / portfolio_size, portfolio_size)
    cons = {"type": "eq", "fun": equality_constraint}
    bnds = tuple([(lower_bound, upper_bound) for x in initial_guess])

    return optimize.minimize(
        objective_function,
        x0=initial_guess,
        args=(mean_returns, covar_returns, risk_free_rate, portfolio_size),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        tol=10**-3,
    )


tickers = st_tags(
    label="## Select Assets:",
    text="Enter a ticker symbol and press enter",
    value=[
        "AAPL",
        "MSFT",
        "GOOG",
        "NFLX",
        "META",
        "AMZN",
    ],
    suggestions=["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "META", "BRK-A", "JNJ"],
    maxtags=8,
)

in1, in2, in3, in4 = st.columns(4)

timeframe = in1.selectbox(
    "Historical Timeframe to Consider",
    options=["2mo", "6mo", "1y", "2y", "5y"],
    index=3,
)

rf = in2.number_input(
    "Risk Free Rate of Return (Annualized)",
    min_value=0.0,
    max_value=10.0,
    value=4.7,
    step=0.5,
    format="%.1f",
)

min_weight = in3.number_input(
    "Minimum Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.05,
)

max_weight = in4.number_input(
    "Maximum Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
)

global dataframe
dataframe = fetch_data(tickers)
(
    simulation,
    returns,
    volatilites,
    allocations,
) = montecarlo(dataframe)
best = simulation.argmax()
winning_sim = pd.DataFrame(
    allocations[best], index=tickers, columns=["Weights"]
).applymap(lambda x: "{:.2%}".format(x))

col1, col2 = st.columns(2)
col1.header("Monte Carlo Portfolio Simulations")
col1.text(
    "- Highest Sharpe Ratio: {0:.2f}\n- Annualized Risk: {1:.2f}%\n- Annualized Est. Return: {2:.2f}%".format(
        simulation[best], volatilites[best], returns[best]
    )
)
col1.table(
    winning_sim,
)

performance = (((dataframe.shift(-1) - dataframe) / dataframe) * 100).dropna()
mean_returns = np.mean(performance.to_numpy(), axis=0)
cov_returns = np.cov(performance.to_numpy(), rowvar=False)

annual_risk_free = rf / 100
r0 = (np.power((1 + annual_risk_free), (1.0 / 360.0)) - 1.0) * 100


result = maximize_sharpe(
    mean_returns, cov_returns, r0, len(tickers), min_weight, max_weight
)
x_optimal = []
x_optimal.append(result.x)
x_opt_array = np.array(x_optimal)
risk = np.matmul((np.matmul(x_opt_array, cov_returns)), np.transpose(x_opt_array))
expected_return = np.matmul(np.array(mean_returns), x_opt_array.T)
annual_risk = np.sqrt(risk * 251)
annual_return = 251 * np.array(expected_return)
max_sharpe = (annual_return - rf) / annual_risk

col2.header("SLSQP - Optimized Portfolio - ✕")
col2.text(
    "- Maximal Sharpe Ratio: {0:.2f}\n- Annualized Risk: {1:.2f}%\n- Annualized Est. Return: {2:.2f}%".format(
        max_sharpe[0][0], annual_risk[0][0], annual_return[0]
    )
)
col2.table(
    pd.DataFrame(x_opt_array.T, index=tickers, columns=["Weights"]).applymap(
        lambda x: "{:.2%}".format(x)
    )
)

plt.figure(figsize=(16, 12), dpi=600)
plt.scatter(volatilites, returns, c=simulation, cmap="plasma")
plt.scatter(
    volatilites[best],
    returns[best],
    s=100,
    c="None",
    edgecolors="white",
)
plt.scatter(
    annual_risk[0][0],
    annual_return[0],
    c="red",
    s=100,
    marker="x",
    edgecolors="white",
)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Risk")
plt.ylabel("Return")
plt.rcParams.update(
    {
        "lines.color": "white",
        "patch.edgecolor": "white",
        "patch.facecolor": "#0E1117",
        "text.color": "black",
        "axes.facecolor": "#0E1117",
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "#0E1117",
        "figure.facecolor": "#0E1117",
        "figure.edgecolor": "#0E1117",
        "savefig.facecolor": "#0E1117",
        "savefig.edgecolor": "#0E1117",
    }
)
st.pyplot(plt)
