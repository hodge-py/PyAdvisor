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
from scipy import stats
from arch import arch_model

plt.style.use('dark_background')


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

    def portfolio_allocation_mv(self,start_date):
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


    def forcast_portfolio_returns_mcs(self,start_date,days_out):
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
            sim_portfolio_value[z] = sim_portfolio_value[z-1] * np.exp((expected_return - 0.5 * expected_volatility ** 2) / 252 + expected_volatility * Wiener_value / np.sqrt(252))

        self._plotMonte("Monte Carlo Simulation Portfolio Returns", sim_portfolio_value,sim_num)

    def forcast_single_stock_mcs(self,start_date,days_out,stock_symbol):
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
        T = days_out  # Days to simulate (1 year)
        num_simulations = 10000  # Number of simulations

        # Step 4: Run Monte Carlo Simulations
        simulated_prices = np.zeros((T, num_simulations))
        simulated_prices[0] = S0

        for t in range(1, T):
            random_shock = np.random.normal(0, 1, num_simulations)
            drift = (mu - 0.5 * sigma ** 2) / 252  # Corrected drift term
            diffusion = sigma.values[0] * random_shock * np.sqrt(1 / 252)  # Corrected diffusion term
            simulated_prices[t] = simulated_prices[t - 1] * np.exp(drift.values + diffusion)

        self._plotMonte(f'Monte Carlo Simulation {stock_symbol} Price',simulated_prices,num_simulations,portfolio_or_stock="stock")

    def _plotMonte(self,title,simulatedP,num_of_sim,portfolio_or_stock='portfolio'):
        """
        Plots the monte carlo simulation when called
        :param title:
        :param simulatedP:
        :param num_of_sim:
        :return:
        """
        if portfolio_or_stock == 'portfolio':
            title_output = 'return'
        elif portfolio_or_stock == 'stock':
            title_output = 'price'
        fig = plt.figure()
        fig.suptitle(title)
        gs = fig.add_gridspec(1, 2, wspace=0)
        (ax1, ax2) = gs.subplots(sharey=True)
        ax1.plot(simulatedP)
        ax1.set_xlabel("Days")
        ax1.set_ylabel(title)
        ax2.hist(simulatedP[-1], orientation='horizontal', bins=int(np.sqrt(num_of_sim)))
        ax2.axhline(np.percentile(simulatedP[-1], 95), color='r')
        ax2.axhline(np.percentile(simulatedP[-1], 50), color='g')
        ax2.axhline(np.percentile(simulatedP[-1], 5), color='yellow')
        plt.show()

        print(f"Median: {np.median(simulatedP[-1])}, Mean: {np.mean(simulatedP[-1])}")
        print(f"95 Percentile {title_output}: {np.percentile(simulatedP[-1], 95)}")
        print(f"50 Percentile {title_output}: {np.percentile(simulatedP[-1], 50)}")
        print(f"5 Percentile {title_output}: {np.percentile(simulatedP[-1], 5)}")

    def generate_sample_portfolio(self,risk='low',include_canada=False):
        if include_canada:
            df_stock = pd.read_csv("StockSymbols.csv")
        else:
            df_stock = pd.read_csv("StockSymbols.csv")
            df_stock = df_stock.drop(columns=['TSX'])

        print(df_stock.head())

    def generateFundamentals(self,stocks, save_to_csv=False):
        valueHold = []
        for z in stocks:
            tmp = [z]
            data = yf.Ticker(z)
            info = data.info
            print(info)
            try:
                tmp.append(self._moneyConvert(int(info['marketCap'])))
            except:
                tmp.append(np.nan)
            try:
                tmp.append(info['trailingPE'])
            except:
                tmp.append(np.nan)
            try:
                tmp.append(info['priceToBook'])
            except:
                tmp.append(np.nan)
            try:
                tmp.append(info['currentRatio'])
            except:
                tmp.append(np.nan)
            try:
                tmp.append(info['debtToEquity'])
            except:
                tmp.append(np.nan)
            try:
                tmp.append(info['returnOnEquity'])
            except:
                tmp.append(np.nan)

            valueHold.append(tmp)


        df_fun = pd.DataFrame(np.array(valueHold), columns=['Ticker',"marketCap",'P/E','P/B','Current Ratio', "Debt to Equity", "Return on Equity"])
        df_fun.set_index('Ticker',inplace=True)
        print(df_fun.to_markdown(tablefmt='github'))

    def options_mcs(self, stock_symbol, start_date, K, T, M, r, sigma, N=10000, option_type="call"):
        """

        :param stock_symbol:
        :param start_date: Start date used for expected returns
        :param K: strike price
        :param T: time to maturity (years)
        :param r: risk-free interest rate
        :param sigma: implied volatility
        :param N: Number of simulations
        :param M: number of time steps (days to expiration)
        :param option_type: "call" or "put"
        :return: None
        """
        data = yf.download(tickers=[stock_symbol], start=start_date, auto_adjust=True).loc[:, 'Close']

        log_returns = np.log(data / data.shift(1)).dropna()

        # Step 2: Estimate Mean and Volatility
        mu = log_returns.mean() * 252  # Annualized return

        dt = T / M  # Time step
        discount_factor = np.exp(-r * T)  # Discount factor for present value
        S0 = data.iloc[-1]  # Current stock price
        # Simulate stock price paths using expected return (mu)
        S = np.zeros((N, M + 1))
        S[:, 0] = S0  # Initial price

        for t in range(1, M + 1):
            Z = np.random.standard_normal(N)  # Random normal variables
            S[:, t] = S[:, t - 1] * np.exp((mu.iloc[0] - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

        # Compute final option payoffs
        S_T = S[:, -1]  # Stock price at expiration
        print(f"Highest simulated stock price: {np.max(S_T)}")
        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0)  # Call option payoff
            number_in_payoffs = np.sum(payoffs > 0) / N * 100
            number_out_payoffs = 100 - number_in_payoffs
        elif option_type == "put":
            payoffs = np.maximum(K - S_T, 0)  # Put option payoff
            number_in_payoffs = np.sum(payoffs < 0) / N * 100
            number_out_payoffs = 100 - number_in_payoffs
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        # Compute option price using discounted expected payoff
        option_price = discount_factor * np.mean(payoffs)

        print(f"Estimated fair price {round(option_price,4)}")
        print(f"Max simulated return: {round(np.max(payoffs)/K * 100,2)}%")
        print(f"Number of simulations in the money %: {round(number_in_payoffs,2)}, out of the money %: {round(number_out_payoffs,2)}")

    def tax_optimization(self):
        pass

    def volatility_options(self,stock_symbol,start_date):
        data = yf.download(tickers=[stock_symbol], start=start_date, auto_adjust=True).loc[:, 'Close']

        log_returns = np.log(data / data.shift(1)).dropna()

        model = arch_model(log_returns*100,vol="GARCH",p=1,q=1,)
        model_fit = model.fit(disp="off")

        forecast = model_fit.forecast(horizon=28)
        garch_volatility = np.sqrt(forecast.variance.iloc[-1]) * np.sqrt(252)

        print(garch_volatility.iloc[-1])

    def forecast_portfolio_lstm(self,plot_stock_forecast=True):
        pass

    def _moneyConvert(self,num):
        if num > 1000000:
            if not num % 1000000:
                return f'${num // 1000000}M'
            return f'${round(num / 1000000, 1)}M'
        return f'${num // 1000}K'


    def standard_deviation_mean_std(self,stocks,start_date,standardize):
        """
        Find the average standard deviation a set of stocks moves.
        :param stocks:
        :param start_date:
        :param standardize:
        :return:
        """

        data = yf.download(tickers=stocks, start=start_date, auto_adjust=True).loc[:, 'Close']

        log_returns = np.log(data / data.shift(1)).dropna()

        # Step 2: Estimate Mean and Volatility
        mu = log_returns.mean() * standardize  # Annualized return
        sigma = log_returns.var() * standardize #Annualized volatility
        sigma = np.sqrt(sigma)

        print(f"mean std: {np.mean(sigma)}, Std: {np.std(sigma)}")
        plt.hist(sigma)
        plt.show()


rb = PyAdvisor([["AAPL",20,200],["META",10,250]]) # Stock symbol, shares, average price

#rb.portfolio_allocation_mv('2024-01-01')
#rb.forcast_portfolio_returns_mcs('2024-01-01',252)
#rb.forcast_single_stock_mcs('2024-01-01',252,"PYPL")
#rb.get_portfolio()
#rb.options_mcs("F",'2024-01-01', 9.35,0.111,28, .05,.3822)
#rb.volatility_options("F",'2024-03-03')
largeCap = ['AXP','PLTR','TMO','ADBE','GS','BX','NOW','VZ','TXN','RTX','QCOM','AMGN','INTU','PGR','RY','AMD','SPGI','UBER','PDD','CAT','BSX','SYK','MUFG','HDB','BLK','DHR','NEE','UNP','PFE','SONY','SCHW','GILD','SNY','TJX','C','TTE','LOW','HON','CMCSA','SHOP','ARM','SBUX','ADP','FI','AMAT','DE','BHP','VRTX','PANW','BA','BUD','BMY','MDT','SPOT','MMC','COP','PLD','NKE','CB','APP','ADI','KKR','UBS','ETN','ANET','LMT','TD','RIO','MELI','MU','UPS','SAN','SO','WELL','LRCX','ICE','AMT','IBN','SMFG','MO','CRWD','KLAC','ENB','WM','INTC','CME','ELV','DUK','RELX','BAM','SHW','EQIX','ABNB','MCO','AON','GEV','AJG','BTI','BP','BN','CI','MDLZ','CTAS','PBR','DASH','RACE','CVS','FTNT','MCK','PH','MMM','TRI','INFY','APO','ORLY','MRVL','GSK','BBVA','HCA','TT','TDG','APH','SE','ECL','ZTS','ITW','MAR','CL','BMO','RSG','CEG','REGN','EPD','PNC','MSTR','MFG']
smallCap = ['LUNR','NEXT','HG','IIPR','GOGL','NVCR','PACS','CEPU','KYMR','BBUC','UPWK','WOR','WMK','STRA','ROCK','CXW','KLIC','OSW','TGI','SSRM','MYRG','RAMP','TRIP','GTX','EXTR','SPB','PLTK','TUYA','WERN','PSEC','SID','SEI','STC','AGIO','IRON','CERT','COCO','HI','VTMX','MESO','CGON','BKE','GNL','GLP','POWL','AZTA','NATL','CLOV','HRMY','ATRC','PHIN','OFG','BLTE','PGNY','VSCO','STNG','SJW','RPD','CWH','SIMO','PWP','HLMN','PBI','VERA','SAND','SBLK','ACLS','CVI','HE','JAMF','GDRX','TWFG','MBIN','CNMD','LZ','MTRN','NIC','JANX','MSDL','EVCM','SUPN','APGE','CMBT','MTTR','LSPD','CASH','CSGS','GB','OMCL','TVTX','LZB','EVT','BCRX','AMR','RVLV','DVAX','DGNX','ARCB','EDN','WVE','CHCO','FIHL','IAS','GTY','TARS','UNFI','PRDO','VRE','DRH','HMN','NWN','IART','ATEC','INSW','TRMD','ADUS','CENX','GLPG','DHT','GBX','ARIS','PAX','AAT','ZD','MBC','MNR','LGIH','FBNC','HIMX','AAPG','BWLP','BTE','PLUS','CNXN','IREN','MCRI','HDL','FL','IDYA','KMT','PLYA','CLBK','AGX','KLG','OI','LKFN','PCT','UFPT','KEN','NTB','AMBP','SEMR','LTC','WKC','ENVX','ADEA','AMPL','BKV','DBD','ACMR','GDYN','FCF']
#rb.standard_deviation_mean_std(smallCap,'2024-01-01',44)