import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.optimize import minimize
from tensorflow.python.ops.gen_array_ops import lower_bound


class MeanVariance:
    def __init__(self,tickers: list[str], timePeriod):
        self.tickers = tickers
        self.timePeriod = timePeriod
        self.df_norm = ""
        self.mean = 0
        self.variance = 0
        self.names = 0
        self.returns = []
        self.volatility = []
        self.args = (0,0)
        self.optimal_weight = []

        self.days = np.busday_count(self.timePeriod,dt.date.today(),weekmask="Mon Tue Wed Thu Fri")

        self._stockHistory()
        self._meanVariance()
        self._calculate_random_portfolio()
        self._minimize_portfolio()
        self._plot_optimal_portfolio()

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

    def _portfolio_mean(self, weight):
        mean_r = self.args[0]
        port_return = np.dot(weight,mean_r)
        return port_return

    def _portfolio_std(self, weight,*args):
        std_r = self.args[1]
        port_std = np.sqrt(np.dot(weight.T,np.dot(std_r,weight)))
        return port_std

    def _minimize_portfolio(self):
        lower_bound = 0
        upper_bound = 1
        bounds = ()
        for i in range(len(self.names)):
            bounds += ((lower_bound,upper_bound),)
        initial_weight = np.ones(len(self.names))/len(self.names)
        print(bounds)
        self.args = (self.mean,self.variance)
        target = 0.12
        constraints = ({"type": 'eq', 'fun': lambda x: self._portfolio_mean(x) - target},
                       {'type': 'eq', 'fun': lambda x: np.sum(x)-1})

        results = minimize(self._portfolio_std, initial_weight, args=self.args, method='SLSQP', bounds=bounds, constraints=constraints)
        self.optimal_weight = results.x
        print(self.optimal_weight)

    def _plot_optimal_portfolio(self):
        optimal_portfolio_plot = np.dot(self.df_norm,self.optimal_weight)
        equal_weight_p = np.dot(self.df_norm,np.ones(len(self.names))/len(self.names))
        plt.plot(optimal_portfolio_plot,label='Optimal Portfolio (15%)')
        plt.plot(equal_weight_p,label='Equal Weight')
        plt.legend(loc='best')
        plt.show()

r = MeanVariance(["ZSP.TO","VFV.TO","XUS.TO"],'2024-01-01')
