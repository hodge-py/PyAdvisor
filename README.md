# PyAdvisor
<p align="center">
  <a href="https://www.python.org">
<img src="https://img.shields.io/badge/Platforms-linux--64,win--64,osx--64-orange.svg?style=flat-square" />
  </a>
</p>

Pyadvisor is a library developed to lended traders a helping hand with *portfolio optimization*, *stock information*, *options*, and *risk analysis*. 

## Examples

### initalization

```python

rb = PyAdvisor([["MSFT",20,417],["META",10,250]]) # initialize the PyAdvisor class and insert portfolio.
#(Stock symbol, shares, average prices)

```

```console

| Symbol   |   Shares |   Average Price |   Initial Value |   Current Value |   Difference |   Weight |
|----------|----------|-----------------|-----------------|-----------------|--------------|----------|
| AAPL     |       20 |             200 |            4000 |          4760.6 |        760.6 |  66.6667 |
| META     |       10 |             250 |            2500 |          6550.5 |       4050.5 |  33.3333 |

```

### Porfolio Optimization Mean Variance

```python

rb.portfolio_allocation('2024-01-01') # start date of how far back the mean historical return should be calculated

```
Output:

```console

| Symbol   |   Weights |   New Share Weight |
|----------|-----------|--------------------|
| AAPL     |    0.2814 |              8.442 |
| META     |    0.7186 |             21.558 |


Expected annual return: 60.3%
Annual volatility: 25.6%
Sharpe Ratio: 2.36
(np.float64(0.6029467652754631), np.float64(0.2555762948674252), np.float64(2.35916545228981))

```

### Forecast Portfolio Returns

```python

rb.forcast_portfolio_returns_mcs('2024-01-01',252) # stard date for return calculation
# and forecast how many days out

```

```console

Median: 8828.008860536578, Mean: 9025.559424303787
95 Percentile Return: 12533.840031710432
50 Percentile Return: 8828.008860536578
5 Percentile Return: 6232.463294590063

```

![image](https://github.com/user-attachments/assets/d4e10cb3-423c-44f5-8a18-222cd20a28ba)

### Forecast a Stock

```python

rb.forcast_single_stock_mcs('2024-01-01',252,"PYPL")

```

```console

Median: 72.56916326545192, Mean: 77.64602181528485
95 Percentile price: 132.49943081758047
50 Percentile price: 72.56916326545192
5 Percentile price: 40.42416434165184

```

![image](https://github.com/user-attachments/assets/5367c1ee-d265-4c1c-83cf-fa5a54a0d378)

### Options Simulation

```python

rb.options_mcs("F",'2024-01-01', 9.35,0.111,28, .05,.3822)

```

```console

Highest simulated stock price: 15.152665407255085
Estimated fair price 0.4257
Max simulated return: 62.06%
Number of simulations in the money %: 44.26, out of the money %: 55.74

```

# Disclaimer
This is free software and is provided as is. The author makes no guarantee that its results are accurate and is not responsible for any losses caused by the use of the code.

This code is provided for educational and research purposes only.

Bugs can be reported as issues.
