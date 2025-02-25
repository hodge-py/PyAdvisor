import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

tickers = ['MSFT',"AAPL","PYPL","TSLA"]
df = yf.download(tickers=tickers, start="2024-01-01")
df_close = df.loc[:,"Close"]

mu = mean_historical_return(df_close)
S = CovarianceShrinkage(df_close).ledoit_wolf()

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(cleaned_weights)
print(ef.portfolio_performance(verbose=True))
