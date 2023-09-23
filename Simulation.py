import math
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas_datareader import data as pdata
import yfinance as yf
yf.pdr_override()
#What is Monte Carlo Simulation?
#Computer Simulations relying on random sampling to obtain results.
def get_data(stocks, start_date, end_date):
    stockData = pdata.get_data_yahoo(stocks, start=start_date, end=end_date)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()

    return meanReturns, covMatrix


# def get_data(stocks, start_date, end_date):
    stockData = pdata.get_data_yahoo(stocks, start_date, end_date) #gets data from yahoo finance
    stockData = stockData['Close'] #takes in only close prices at the end of the day. 
    returns = stockData.pct_change()
    meanRe = returns.mean()
    covarianceMatrix = returns.cov()
    return meanRe, covarianceMatrix

stockList = ["AMZN", 'MELI', 'NVDA', 'DIS', 'AAPL', 'TSLA', 'META', 'GOOGL', 'AMD']
#stockList = ['AMZN', 'NVDA']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stockList, startDate, endDate)

#print(meanReturns)

#for symbol, mean in meanReturns.items():
   # print(f"Mean return for {symbol}: {mean}")

#for symbol, cov in covMatrix.items():
  #  print(f"Covariance matrix for {symbol}:\n{cov}")


weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)
print(f"weights = {weights}")

#Monte Carlo Simulations
numOfSims = 100
numOfDays = 100 #in days

#array to store information
meanMatrix = np.full(shape=(numOfDays, len(weights)), fill_value=meanReturns)
meanMatrix = meanMatrix.T #Transpose for the method
initialfolioValue = 10000
portfolio_sims = np.full(shape=(numOfDays, numOfSims), fill_value=0.0)
for m in range(0, numOfSims):
    #do montecarlo simulation
    x = np.random.normal(size = (numOfDays, len(weights)))
    y = np.linalg.cholesky(covMatrix) #finds lower triangle for a cholesky decomposition from the covariance matrix
    dailyReturns = meanMatrix + np.inner(y, x)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialfolioValue
plt.plot(portfolio_sims)
plt.ylabel('Porfolio Value($)') 
plt.xlabel('Days')   
plt.title('Monte Carlo Simulation of Stock Portfolio')
plt.show()