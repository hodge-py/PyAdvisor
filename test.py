import PyTrader

pyT = PyTrader.PyTrader()
pyT.stockScreener(["MSFT","FINV", "TSLA"])

pyT.generateTechnicalReport(["MSFT","FINV","VRRM"])