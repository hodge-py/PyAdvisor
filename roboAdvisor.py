import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RoboAdvisor:
    def __init__(self,tickers: list[str], timePeriod):
        self.tickers = tickers
        self.timePeriod = timePeriod


    def riskValue(self):
        pass



r = RoboAdvisor(['MSFT','AAPL'])
