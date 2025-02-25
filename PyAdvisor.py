import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier



class PyAdvisor:

    def __init__(self,portfolio):
        self.portfolio = pd.DataFrame()
        self._initial_portfolio(portfolio)

    def _initial_portfolio(self, portfolio):
        df = pd.DataFrame(portfolio,columns=['Symbol','Shares','Average Price'])
        df.index = df['Symbol']
        df.drop('Symbol',axis=1,inplace=True)
        df['Initial Value'] = df['Shares'] * df['Average Price']
        df_price = yf.download(tickers=list(df.index.values), period='1d',auto_adjust=True)
        df_close = df_price.loc[:, "Close"]
        holder = []
        arr = df_close.values.reshape(-1,1)
        arr = pd.DataFrame(arr)
        df['Current Value'] = arr.iloc[:,0].values*df['Shares']
        df['Difference'] = df['Current Value'] - df['Initial Value']
        df['Weight'] = df['Shares'] / np.sum(df['Shares']) * 100
        print(df.to_markdown(tablefmt='github'))
        self.portfolio = df

    def set_portfolio(self,portfolio):
        df = pd.DataFrame(portfolio, columns=['Symbol', 'Shares', 'Average Price'])
        df.index = df['Symbol']
        df.drop('Symbol', axis=1, inplace=True)
        df['Initial Value'] = df['Shares'] * df['Average Price']
        df_price = yf.download(tickers=list(df.index.values), period='1d', auto_adjust=True)
        df_close = df_price.loc[:, "Close"]
        holder = []
        arr = df_close.values.reshape(-1, 1)
        arr = pd.DataFrame(arr)
        df['Current Value'] = arr.iloc[:, 0].values * df['Shares']
        df['Difference'] = df['Current Value'] - df['Initial Value']
        df['Weight'] = df['Shares'] / np.sum(df['Shares']) * 100
        print(df.to_markdown(tablefmt='github'))
        self.portfolio = df

    def portfolio_allocation(self,start_date):
        self._meanVariance(start_date)

    def _meanVariance(self,start_date):
        df_prices = yf.download(tickers=list(self.portfolio.index.values), start=start_date)
        df_final = df_prices.loc[:, "Close"]

        mu = mean_historical_return(df_final)
        S = CovarianceShrinkage(df_final).ledoit_wolf()

        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()

        cleaned_weights = ef.clean_weights()
        tmplist = []
        for z in cleaned_weights:
            tmplist.append([z,cleaned_weights[z]])
        dataF = pd.DataFrame(tmplist, columns=['Symbol',"Weights"])
        dataF.index = dataF['Symbol']
        dataF.drop('Symbol',axis=1,inplace=True)
        dataF['New Share Weight'] = dataF['Weights'].values * np.sum(self.portfolio['Shares'].values)
        print(dataF.to_markdown(tablefmt='github'))
        print("\n")
        print(ef.portfolio_performance(verbose=True))

    def Rebalncing(self):
        pass

    def tax_optimization(self):
        pass


rb = PyAdvisor([["MSFT",20,417],["TSLA",10,250]])

rb.portfolio_allocation('2024-01-01')