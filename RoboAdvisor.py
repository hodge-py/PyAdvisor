import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


class PyAdvisor:

    def __init__(self):
        self.portfolio = 0

    def set_portfolio(self, portfolio):
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

    def portfolio_allocation(self):
        pass

    def Rebalncing(self):
        pass

    def tax_optimization(self):
        pass


rb = PyAdvisor()
rb.set_portfolio([["MSFT",20,417],["TSLA",10,250]])