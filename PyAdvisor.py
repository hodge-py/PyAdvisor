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
        """
        initializes the portfolio variable
        :param portfolio:
        """
        self.portfolio = pd.DataFrame()
        self._initial_portfolio(portfolio)

    def _initial_portfolio(self, portfolio):
        """
        Sets the initial portfolio for use.

        :param portfolio: 2d array of [[Symbol, Shares, Average Price]]
        :return: None
        """
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
        """
        Sets a new portfolio for use.
        :param portfolio:
        :return:
        """
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

    def get_portfolio(self):
        """
        Prints the current portfolio.
        :return:
        """
        print(self.portfolio.to_markdown(tablefmt='github'))

    def portfolio_allocation(self,start_date):
        self._meanVariance(start_date)

    def _meanVariance(self,start_date):
        """

        :param start_date: yyyy-mm-dd, starting date for the expected return
        :return:
        """
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
        """

        :param start_date: yyyy-mm-dd format, starting date for history of close price.
        :param days_out: Amount of days to forecast out.
        :return:
        """
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

    def forcast_single_stock(self,start_date,days_out, stock_symbol):
        """

        :param start_date: yyyy-mm-dd format, starting date for history of close price.
        :param days_out: amount of days to forecast out.
        :param stock_symbol:
        :return:
        """
        data = yf.download(tickers=[stock_symbol], start=start_date, auto_adjust=True).loc[:,'Close']
        returns = np.log(data / data.shift(1)).dropna()

        log_returns = np.log(data / data.shift(1)).dropna()

        # Step 2: Estimate Mean and Volatility
        mu = log_returns.mean() * 252  # Annualized return
        sigma = log_returns.std() * np.sqrt(252)  # Annualized volatility

        # Step 3: Monte Carlo Simulation Parameters
        S0 = data.iloc[-1]  # Current stock price
        T = 252  # Days to simulate (1 year)
        num_simulations = 10000  # Number of simulations

        # Step 4: Run Monte Carlo Simulations
        simulated_prices = np.zeros((T, num_simulations))
        simulated_prices[0] = S0

        for t in range(1, T):
            random_shock = np.random.normal(0, 1, num_simulations)
            drift = (mu - 0.5 * sigma ** 2) / 252  # Corrected drift term
            diffusion = sigma.values[0] * random_shock * np.sqrt(1 / 252)  # Corrected diffusion term
            simulated_prices[t] = simulated_prices[t - 1] * np.exp(drift.values + diffusion)

        fig = plt.figure()
        fig.suptitle(f'Monte Carlo Simulation {stock_symbol} Returns')
        gs = fig.add_gridspec(1, 2, wspace=0)
        (ax1, ax2) = gs.subplots(sharey=True)
        ax1.plot(simulated_prices)
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Portfolio Value")
        ax2.hist(simulated_prices[-1], orientation='horizontal',bins=int(np.sqrt(num_simulations)))
        ax2.axhline(np.percentile(simulated_prices[-1], 95), color='r')
        ax2.axhline(np.percentile(simulated_prices[-1], 50), color='g')
        ax2.axhline(np.percentile(simulated_prices[-1], 5), color='black')
        plt.show()

    def generate_sample_portfolio(self,risk='low'):
        pass

    def tax_optimization(self):
        pass


rb = PyAdvisor([["MSFT",20,417],["META",10,250]])

#rb.portfolio_allocation('2024-01-01')
#rb.forcast_portfolio_returns('2024-01-01',252)
rb.forcast_single_stock('2024-01-01',252,"MSFT")
#rb.get_portfolio()