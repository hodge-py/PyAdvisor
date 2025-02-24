import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


class RoboAdvisor:
    def __init__(self,tickers: list[str], timePeriod):
        self.tickers = tickers
        self.timePeriod = timePeriod
        self.df_norm = ""
        self.mean = 0
        self.variance = 0
        self.names = 0
        self.returns = []
        self.volatility = []

        self.days = np.busday_count(self.timePeriod,dt.date.today(),weekmask="Mon Tue Wed Thu Fri")

        self._stockHistory()
        self._meanVariance()
        self._calculate_random_portfolio()

    def _stockHistory(self):
        df = yf.download(tickers=self.tickers, start=self.timePeriod)
        df_close = df.loc[:,"Close"]
        self.names = df_close.columns.tolist()
        self.df_norm = self._Normalization(df_close)
        self.df_norm.plot()
        plt.show()

    def _Normalization(self,data):
        priceNorm = data / data.iloc[0,:] * 100
        return priceNorm

    def _meanVariance(self):
        data_r = np.log(self.df_norm).diff().dropna()
        return_data = data_r * self.days
        self.mean = return_data.mean().values
        self.variance = data_r.cov() * self.days
        print(self.mean, self.variance)

    def _calculate_random_portfolio(self):
        for x in range(1000):
            weights_random = np.random.random(len(self.names))
            weights_norm = weights_random / np.sum(weights_random)
            randomPortfolioReturns = np.dot(weights_norm,self.mean)
            self.returns.append(randomPortfolioReturns)
            randomVariance = np.dot(np.dot(weights_norm.T,self.variance),weights_norm)
            deviation = np.sqrt(randomVariance)
            self.volatility.append(deviation)

        plt.scatter(self.volatility, self.returns)
        plt.show()



r = RoboAdvisor(['MSFT','AAPL','ADBE','META'],'2024-01-01')
