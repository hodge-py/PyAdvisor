import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PyTrader:

    def __init__(self):
        pass

    def generateValueReport(self,stocks):
        pass

    def stockScreener(self, stocks, PB = 5, PE = 5):
        UnderValued = []
        for z in stocks:
            y = yf.Ticker(z)
            closePrice = y.history(period="1mo")['Close'].iloc[[-1]]
            priceBook = closePrice.iloc[0] / (y.quarterly_balance_sheet.iloc[:,0]['Tangible Book Value'] / y.quarterly_balance_sheet.iloc[:,0]['Share Issued'])

            if priceBook < PB:
                UnderValued.append(z)


        print("UnderValued Stocks: " + str(UnderValued))


    def generateTechnicalReport(self, stocks, shortWin = 50, longWin = 200):
        for z in stocks:
            data = yf.Ticker(z)
            data = data.history(period="1y")
            data["SMA_50"] = data["Close"].rolling(window=shortWin).mean()
            data["SMA_200"] = data["Close"].rolling(window=longWin).mean()

            # Plot the data
            plt.figure(figsize=(12, 6))
            plt.plot(data["Close"], label="Stock Price", alpha=0.6)
            plt.plot(data["SMA_50"], label=f"{shortWin}-Day SMA", linestyle="dashed")
            plt.plot(data["SMA_200"], label=f"{longWin}-Day SMA", linestyle="dotted")
            plt.legend()
            plt.title(f"{z} Moving Average Crossover")
            plt.show()

            if (data["SMA_50"].iloc[-1] > data["SMA_200"].iloc[-1]) and (abs((data["SMA_50"].iloc[-1] - data["SMA_200"].iloc[-1]) / data["SMA_50"].iloc[-1]) <= 0.01):
                print("Buy " + str(z))
                print(abs((data["SMA_50"].iloc[-1] - data["SMA_200"].iloc[-1]) / data["SMA_50"].iloc[-1]))
            elif ((data["SMA_50"].iloc[-1] < data["SMA_200"].iloc[-1]) and (abs((data["SMA_50"].iloc[-1] - data["SMA_200"].iloc[-1]) / data["SMA_50"].iloc[-1]) <= 0.01)):
                print("Sell " + str(z))
            else:
                print("Hold/Do Nothing " + str(z))
