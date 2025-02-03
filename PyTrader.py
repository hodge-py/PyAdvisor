import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
from ta.volatility import BollingerBands

class PyTrader:

    def __init__(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

    def generateValueReport(self,stocks,save_as_csv=False):
        arr = []
        for stock in stocks:
            y = yf.Ticker(stock)
            try:
                closePrice = y.history(period="1mo")['Close'].iloc[[-1]]
            except:
                closePrice = 0

            try:
                financialsEPS = y.info['trailingEps']
                priceEarnings = closePrice.iloc[0] / financialsEPS
            except:
                priceEarnings = None
            try:
                quarters = y.quarterly_balance_sheet.iloc[:, 0]
                priceBook = closePrice.iloc[0] / (quarters['Tangible Book Value'] /
                                              quarters['Share Issued'])
            except:
                priceBook = None

            try:
                financialPeg = y.info['trailingPegRatio']
            except:
                financialPeg = None

            try:
                debtEquity = y.info['debtToEquity']
            except:
                debtEquity = None
                #financialPeg = None
            try:
                quickly = y.info['quickRatio']
            except:
                quickly = None

            links = f"https://finance.yahoo.com/quote/{stock}"
            arr.append([stock,priceBook,priceEarnings,financialPeg,debtEquity,quickly,links])

        df = pd.DataFrame(arr, columns=['Stock','Price/Book','Price/Earnings',"Trailing Peg Ratio", "DebtToEquity", "QuickRatio", "Link"])
        df = df.sort_values(['Price/Book', 'Price/Earnings'], ascending=[False,False])
        print(df)
        if save_as_csv:
            df.to_csv("stocks.csv", index=True)


    def stockScreener(self, stocks, PB = 5, PE = 5):
        UnderValued = []
        for z in stocks:
            y = yf.Ticker(z)
            closePrice = y.history(period="1mo")['Close'].iloc[[-1]]
            priceBook = closePrice.iloc[0] / (y.quarterly_balance_sheet.iloc[:,0]['Tangible Book Value'] / y.quarterly_balance_sheet.iloc[:,0]['Share Issued'])

            if priceBook < PB:
                UnderValued.append(z)

        print("UnderValued Stocks: " + str(UnderValued))

        # Saves undervalued stocks to clip board to use for generate ValueReport
        def addToClipBoard(text):
            command = 'echo ' + text.strip() + '| clip'
            os.system(command)

        addToClipBoard(str(UnderValued))

        return UnderValued

    def generateSMAReport(self, stocks, shortWin = 50, longWin = 200, save_as_csv=False):
        allTechnicals = []
        for z in stocks:
            data = yf.Ticker(z)
            data = data.history(period="1y")
            data["SMA_50"] = data["Close"].rolling(window=shortWin).mean()
            data["SMA_200"] = data["Close"].rolling(window=longWin).mean()
            #indicator_bb = BollingerBands(close=data["Close"], window=50,window_dev=2)

            sma = ''
            try:
                if (data["SMA_50"].iloc[-1] > data["SMA_200"].iloc[-1]) and (0 <= abs((data["SMA_50"].iloc[-1] - data["SMA_200"].iloc[-1]) / data["SMA_50"].iloc[-1]) <= .01):
                    sma = "Examine"
                    self._generateSMAGraph(data, z,shortWin, longWin)
                    print(abs((data["SMA_50"].iloc[-1] - data["SMA_200"].iloc[-1]) / data["SMA_50"].iloc[-1]))
                elif (data["SMA_50"].iloc[-1] < data["SMA_200"].iloc[-1]) and (0 <= abs((data["SMA_50"].iloc[-1] - data["SMA_200"].iloc[-1]) / data["SMA_50"].iloc[-1]) <= 0.01):
                    sma = "Examine"
                    #print("Sell " + str(z))
                    self._generateSMAGraph(data, z,shortWin, longWin)
                else:
                    sma = "Hold/Do Nothing"
                    #print("Hold/Do Nothing " + str(z))

            except:
                sma = None


            allTechnicals.append([z,sma])

        df2 = pd.DataFrame(allTechnicals,columns=['Ticker', 'SMA'])

        print(df2)

        if save_as_csv:
            df2.to_csv("stocksTechnical.csv", index=True)



    def _generateSMAGraph(self,data,z,shortSma,longSma):
        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(data["Close"], label="Stock Price", alpha=0.6)
        plt.plot(data["SMA_50"], label=f"{shortSma}-Day SMA", linestyle="dashed",color="red")
        plt.plot(data["SMA_200"], label=f"{longSma}-Day SMA", linestyle="dotted", color="green")
        plt.legend()
        plt.title(f"{z} Moving Average Crossover")
        plt.show()


    def combineTables(self):
        pass