# stocktrading_algo

The results from this ongoing project can be seen at www.pairsstocktrading.com

The pairstrading_daily.py algorithm is executed every business day after the stock market closes, and it updates files for the website listed above. It takes in a list of all stocks and their adjusted close prices over a training period 't', tests for pair correlation greater than 85%, and then runs a test for cointegration and stationarity with a pvalue of 0.01. The pairs that pass all of these tests are then ranked by their volatility (most volatile are best), and the top 'n' are chosen to be monitered for the upcoming month. Then, for every day of the month, the pairs are checked for buy/sell indicators and acted upon. 

The pairstrading_backtest.py algorithm is the exact same code, except instead of running every day, it can be given a range of historical dates between 2010 and the present to test different strategies. 

This is part of a larger project that I hope to keep expanding in the future.
