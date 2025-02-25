import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from contourpy import as_z_interp
from matplotlib import gridspec
from matplotlib.pyplot import xlabel
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
        df_prices = yf.download(tickers=list(self.portfolio.index.values), start=start_date,auto_adjust=True)
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


    def forcast_portfolio_returns(self,start_date,days_out):
        data = yf.download(tickers=list(self.portfolio.index.values), start=start_date,auto_adjust=True).loc[:,'Close']
        returns = np.log(data / data.shift(1)).dropna()

        weight = self.portfolio['Weight'].values
        mean = returns.mean() * 252
        variance = returns.cov() * 252

        expected_return = np.sum((weight/100)*mean)
        expected_volatility = np.sqrt(np.dot((weight/100).T,np.dot(variance,(weight/100))))

        sim_num = 10000
        time_horizon = days_out
        initial_value = np.sum(self.portfolio['Initial Value'])

        sim_portfolio_value = np.zeros((time_horizon, sim_num))
        sim_portfolio_value[0] = initial_value

        for z in range(1, time_horizon):
            Wiener_value = np.random.normal(0,1,sim_num)
            sim_portfolio_value[z] = sim_portfolio_value[z-1] * np.exp((expected_return - 0.5 * expected_volatility ** 2) / days_out + expected_volatility * Wiener_value / np.sqrt(252))

        fig = plt.figure()
        fig.suptitle('Monte Carlo Simulation Portfolio Returns')
        gs = fig.add_gridspec(1,2, wspace=0)
        (ax1,ax2) = gs.subplots(sharey=True)
        ax1.plot(sim_portfolio_value)
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Portfolio Value")
        ax2.hist(sim_portfolio_value[-1],orientation='horizontal',bins=int(np.sqrt(sim_num)))
        ax2.axhline(np.percentile(sim_portfolio_value[-1],95),color='r')
        ax2.axhline(np.percentile(sim_portfolio_value[-1],50),color='g')
        ax2.axhline(np.percentile(sim_portfolio_value[-1],5),color='black')
        plt.show()

        print(f"Median: {np.median(sim_portfolio_value[-1])}, Mean: {np.mean(sim_portfolio_value[-1])}")
        print(f"95 Percentile Return: {np.percentile(sim_portfolio_value[-1],95)}")
        print(f"50 Percentile Return: {np.percentile(sim_portfolio_value[-1],50)}")
        print(f"5 Percentile Return: {np.percentile(sim_portfolio_value[-1],5)}")

    def forcast_single_stock(self,start_date,days_out):
        pass

    def generate_sample_portfolio(self,risk='low'):
        pass

    def tax_optimization(self):
        pass


rb = PyAdvisor([["MSFT",20,417],["META",10,250]])

#rb.portfolio_allocation('2024-01-01')
rb.forcast_portfolio_returns('2024-01-01',252)