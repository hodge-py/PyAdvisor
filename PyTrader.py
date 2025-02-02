import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = yf.Ticker("TSLA")
plots = data.history(period="max")
print(data.quarterly_financials.iloc[:,3])
plt.plot(plots.index.values, plots.Close)
plt.show()
