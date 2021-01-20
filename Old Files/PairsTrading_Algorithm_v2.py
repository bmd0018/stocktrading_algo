#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:05:17 2020

@author: brettdavis
"""

from bs4 import BeautifulSoup as bs
import requests
import pickle
import yahoo_fin.stock_info as yf
import yfinance as yfin
import datetime as dt
import dateutil.relativedelta as rel_delta
import os
import time
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import tensorflow as tf
import math

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint,adfuller

import progressbar as pgb
from itertools import repeat

import signal
from contextlib import contextmanager

# df = pd.read_csv('Ticker Lists/AllTicks.csv')
# tickers = df['Symbol']


def get_r1000_tickers():
    url = 'https://en.wikipedia.org/wiki/Russell_1000_Index'
    r = requests.get(url)
    
    soup = bs(r.content,'html.parser')
    
    table = soup.find_all('table',class_='wikitable sortable')
    
    tickers = []
    for row in table[0].find_all('tr')[1:]:
        ticker = row.find_all('td')[1].text[:-1]
        tickers.append(ticker)
    
    with open('r1000ticks.pickle','wb') as f:
        pickle.dump(tickers,f)
    
    return tickers

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



def get_10y1d_data(reload_r1000=False):
    if reload_r1000:
        tickers = get_r1000_tickers()
    else:
        with open('r1000ticks.pickle','rb') as f:
            tickers = pickle.load(f)
      
    df_price = pd.DataFrame()  
    sdf = yf.get_data('GOOG',start_date='2010-01-01 00:00:00')
    df_price['GOOG'] = sdf['adjclose']        
    
    for tick in tickers:
        
        try:
            with time_limit(10):
            
                try:
                    sdf = yf.get_data(tick,start_date='2010-01-01 00:00:00')
                    df_price[tick] = sdf['adjclose']
                    
                    print('Finished: {}'.format(tick))
                except:
                    pass
            
        except TimeoutException as e:
            print ('Timed out!')
            
    corrmatrix = df_price.corr()
    
    with open('Pairs Trading Stock Data/correlation_matrix_20y1d.pickle','wb') as f:
        pickle.dump(corrmatrix,f)
    with open('Pairs Trading Stock Data/stock_10y1d_price.pickle','wb') as f:
        pickle.dump(df_price,f)


def get_7dmin_data(reload_r1000=False):
    if reload_r1000:
        tickers = get_r1000_tickers()
    else:
        with open('r1000ticks.pickle','rb') as f:
            tickers = pickle.load(f)
    
    df_change = pd.DataFrame()
    df_price = pd.DataFrame()
    for tick in tickers:
        try: 
            yftick = yfin.Ticker(tick)
            sdf = yftick.history(period='7d',interval='1m')
            df_price[tick] = sdf['Close']
            df_change[tick] = (sdf['Close'] - sdf['Close'].shift(-1))/sdf['Close']
            
            print('Added {}'.format(tick))
        except:
            print('No data found for {}'.format(tick)) 
            
    corrmatrix = df_price.corr()
    
    with open('Pairs Trading Stock Data/correlation_matrix.pickle','wb') as f:
        pickle.dump(corrmatrix,f)
    with open('Pairs Trading Stock Data/stock_7dmin_change.pickle','wb') as f:
        pickle.dump(df_change,f)
    with open('Pairs Trading Stock Data/stock_7dmin_price.pickle','wb') as f:
        pickle.dump(df_price,f)


def calc_ratio_trends(pairs,train):
    meanlist = []
    stdlist = []
    ratio_trend = []
    ratio_intercept = []

    for row in range(len(pairs)):
        
        ratios = pd.DataFrame(columns=['days','ratio'])
    
        tick1 = pairs['row'][row]
        tick2 = pairs['col'][row]
        
        meanlist.append((train[tick2]/train[tick1]).mean())
        stdlist.append((train[tick2]/train[tick1]).std())
        
        ratios['ratio'] = (train[tick2]/train[tick1])
        ratios['days'] = (train.index-train.index[0]).days
        
        ratios.dropna(inplace=True)

        model = linear_model.LinearRegression()
        reg = model.fit(ratios[['days']],ratios[['ratio']])
        ratio_trend.append(reg.coef_[0][0])
        ratio_intercept.append(reg.intercept_[0])
        
        # plt.figure()
        # plt.plot(ratios['days'],ratios['ratio'])
        # plt.plot(ratios['days'],ratios['days']*ratio_trend[row]+ratio_intercept[row])
        
        
    pairs['avgRatio'] = meanlist
    pairs['stdRatio'] = stdlist
    pairs['slope'] = ratio_trend
    pairs['intercept'] = ratio_intercept

    return pairs  



def check_for_stationarity(x, cutoff=0.01):
    # null hypothesis in adfuller test is non-stationarity
    # we need a p-value lower than the hypothesis to reject (implies stationarity)
    
    pvalue = adfuller(x)[1]
    
    if pvalue < cutoff:
        # print('p-value is: {}. Series is likely STATIONARY.'.format(pvalue))
        return True
    else:
        # print('p-value is: {}. Series is likely NON-STATIONARY.'.format(pvalue))
        return False


    
def extract_correlation_pairs(df,act_day,train_length=6,test_forward=1): # train length in months
    
    train = df.loc[(act_day+rel_delta.relativedelta(months=-train_length)):act_day]     
    test = df.loc[act_day:(act_day+rel_delta.relativedelta(months=test_forward))] 
    
    # get pairs with 80% correlation or better
    cmat = train.corr()
    
    dataCorr = cmat.stack().reset_index()
    
    # filter out exact matches
    dataCorr = dataCorr[dataCorr['level_0'].astype(str)!=dataCorr['level_1'].astype(str)]
     
    # filtering out lower/upper triangular duplicates 
    dataCorr['ordered-cols'] = dataCorr.apply(lambda x: '-'.join(sorted([x['level_0'],x['level_1']])),axis=1)
    dataCorr = dataCorr.drop_duplicates(['ordered-cols'])
    dataCorr.drop(['ordered-cols'], axis=1, inplace=True)
    
    good = dataCorr[((dataCorr.iloc[:,2]>=0.95) & (dataCorr.iloc[:,2]<0.99))]
    
    t1 = list(good['level_0'])
    t2 = list(good['level_1'])
    vals = good.iloc[:,2]
    
    pairs = pd.DataFrame({'row':t1,'col':t2,'corr':vals})
    
    print('There are {} correlated pairs.'.format(len(pairs)))
    print('')

    return pairs,train,test


def test_pair_cointegration(args):
    
    t1 = args[0]
    t2 = args[1]
    train = args[2]

    if any(train[t1]<0) or any(train[t2]<0):
        pvalue = np.nan
        
    else:

        logvals = np.array([np.array(np.log10(train[t1])),np.array(np.log10(train[t2]))]).transpose()
        
        logvals = logvals[~np.isnan(logvals).any(axis=1)]
        
        if len(logvals) >= 50:
        
            _,pvalue,_ = coint(logvals[:,0],logvals[:,1])
        else:
            pvalue = np.nan
              
    return pvalue

                
def pair_cointegration_stats(pairs,df,act_day,train_length=3):
    
    # widgets = [pgb.FormatLabel('Calculating Pair Stats: '),pgb.Bar('*'),pgb.Timer(format= '  Timer:   %(elapsed)s'),'   ',pgb.ETA()] 
    # bar = pgb.ProgressBar(max_value=len(pairs),widgets=widgets)
    
    train = df.loc[(act_day+rel_delta.relativedelta(months=-train_length)):act_day] 
    
    fpairs = pd.DataFrame(columns=['Tick1','Tick2','beta','zavg','zstd','zmax','success'])
    for row in range(len(pairs)):
        # bar.update(row)
        logvals = pd.DataFrame(columns=('Tick1','Tick2'))
        
        t1 = pairs['row'][row]
        t2 = pairs['col'][row]
        
        logvals['Tick1'] = np.log10(train[t1])
        logvals['Tick2'] = np.log10(train[t2])
        
        logvals.dropna(inplace=True)
        
        if len(logvals) >= 10:
    
            add1s = sm.add_constant(logvals)
            results = sm.OLS(add1s['Tick2'],add1s.drop(columns='Tick2')).fit()
            
            B = (results.params['Tick1'])  
            z = logvals['Tick2'] - B*logvals['Tick1'] 
            pcheck = check_for_stationarity(z,0.01)
        else:
            pcheck = False
        
        if pcheck:
            zmean = z.mean()
            zstd = z.std()
            zmax = abs(z-z.mean()).max()
            
            pos = 0
            success = 0
            for ii in range(len(z))[1:]:
                
                if pos == 0:
                    if z[ii] > (zmean+zstd):
                        pos = 1
                    elif z[ii] < (zmean-zstd):
                        pos = -1
                    else:
                        pass
                
                elif pos == 1:
                    if (abs(z[ii-1]) > abs(zmean) > abs(z[ii])):
                        pos = 0
                        success += 1
                        
                elif pos == -1:
                    if (abs(z[ii-1]) < abs(zmean) < abs(z[ii])):
                        pos = 0
                        success += 1
                        
                else:
                    pass# print('Error!')
                    
            
                
            fpairs = fpairs.append({'Tick1':t1,'Tick2':t2,'beta':B,'zavg':zmean,'zstd':zstd,
                                        'zmax':zmax,'success':success},ignore_index=True)
            
    fpairs = fpairs.convert_dtypes()
    fpairs = fpairs.nlargest(200,'success',keep='all')
    fpairs.reset_index(drop=True,inplace=True)
    
    print('')
    print('There are {} final fpairs.'.format(len(fpairs)))
    print('')
    
    return fpairs


def check_inactive_pairs(fpairs,daily_data,position,active_trades,day,amount,cash):
    for row in range(len(fpairs)):
        
        try:
            if position[row] == 0:
                t1 = fpairs['Tick1'][row]
                t2 = fpairs['Tick2'][row]
                name = t1+'_'+t2
                b = fpairs['beta'][row]
                
                mean = fpairs['zavg'][row]
                std = fpairs['zstd'][row]
                zval = np.log10(daily_data[t2]) - b * np.log10(daily_data[t1])
                
                zscore = (zval-mean)/std            
                
                data = [t1,t2,b,mean,std]
                
                if name in list(active_trades):
                    pass
                
                else:
                    if (-0.2 < zscore < 0.2):
                        if cash >= amount*2:
                            active_trades[name] = [pd.DataFrame({'Date':[day],'Zscore':[zscore],'T1 Price':[daily_data[t1]],'T2 Price':[daily_data[t2]],
                                                          'T1 Shares':[amount/daily_data[t1]],'T2 Shares':[amount/daily_data[t2]],
                                                          'T1 Net':[(-amount)],'T2 Net':[(-amount)],'Overall Net':[(-2*amount)],
                                                          'Position':[0]}),data,0]
                            position[row] = 1
                            cash += -amount*2
        
        except:
            print('Inactive error with pair.')
            
                
    return active_trades,position,cash


def trade_active_pairs(fpairs,daily_data,position,active_trades,trade_log,day,amount,cash):
    active_keys = list(active_trades)
    for row in range(len(active_keys)):
        
        try:
            name = active_keys[row]
            
            t1 = active_trades[name][1][0]
            t2 = active_trades[name][1][1]
            b = active_trades[name][1][2]
            mean = active_trades[name][1][3]
            std = active_trades[name][1][4]
            pos = active_trades[name][0]['Position'].iloc[-1]
            
            
            zval = np.log10(daily_data[t2]) - b * np.log10(daily_data[t1])
            
            zscore = (zval-mean)/std
            
            # check if I have a current position for pair
            if pos == 0:
                
                # z rises too high, buy $100 tick1 stock, sell $100 tick2
                if (zscore > 1) and (zscore < 8):
                    active_trades[name][0] = active_trades[name][0].append({
                        'Date':day,
                        'Zscore':zscore,
                        'T1 Price':daily_data[t1],
                        'T2 Price':daily_data[t2],
                        'T1 Shares':amount/daily_data[t1],
                        'T2 Shares':-amount/daily_data[t2],
                        'T1 Net':(-amount),
                        'T2 Net':(amount),
                        'Overall Net':0,
                        'Position':1},
                        ignore_index=True)
                    
                    
                # ratio falls too low, sell $100 tick1, buy $100 tick2
                elif (zscore < -1) and (zscore > -8):
                    active_trades[name][0] = active_trades[name][0].append({
                        'Date':day,
                        'Zscore':zscore,
                        'T1 Price':daily_data[t1],
                        'T2 Price':daily_data[t2],
                        'T1 Shares':-amount/daily_data[t1],
                        'T2 Shares':amount/daily_data[t2],
                        'T1 Net':(amount),
                        'T2 Net':(-amount),
                        'Overall Net':0,
                        'Position':-1},
                        ignore_index=True)
                    
                    
                # if neither happens, do nothing
                else:
                    pass
            
            # if I do have a current position
            elif abs(pos) == 1:
                    
                # is pos is 1, sell T1 and buy T2
                if pos == 1:
                    if (zscore < 0.2):
                        
                        shares = [sum(active_trades[name][0]['T1 Shares']),sum(active_trades[name][0]['T2 Shares'])]
                        shares100 = [amount/daily_data[t1],amount/daily_data[t2]]
                        
                        t1_settle = shares100[0]-shares[0]
                        t2_settle = shares100[1]-shares[1]
                    
                        active_trades[name][0] = active_trades[name][0].append({
                            'Date':day,
                            'Zscore':zscore,
                            'T1 Price':daily_data[t1],
                            'T2 Price':daily_data[t2],
                            'T1 Shares':t1_settle,
                            'T2 Shares':t2_settle,
                            'T1 Net':-t1_settle*daily_data[t1],
                            'T2 Net':-t2_settle*daily_data[t2],
                            'Overall Net':-t1_settle*daily_data[t1]-t2_settle*daily_data[t2],
                            'Position':0},
                            ignore_index=True)
                        
                        # add one success
                        active_trades[name][2] += 1
                        
                        cash += -t1_settle*daily_data[t1]-t2_settle*daily_data[t2]
                        
                        
                # is pos is -1, buy T1 and sell T2
                elif pos == -1:
                    if (zscore > -0.2):
                        
                        shares = [sum(active_trades[name][0]['T1 Shares']),sum(active_trades[name][0]['T2 Shares'])]
                        shares100 = [amount/daily_data[t1],amount/daily_data[t2]]
                        
                        t1_settle = shares100[0]-shares[0]
                        t2_settle = shares100[1]-shares[1]
                    
                        active_trades[name][0] = active_trades[name][0].append({
                            'Date':day,
                            'Zscore':zscore,
                            'T1 Price':daily_data[t1],
                            'T2 Price':daily_data[t2],
                            'T1 Shares':t1_settle,
                            'T2 Shares':t2_settle,
                            'T1 Net':-t1_settle*daily_data[t1],
                            'T2 Net':-t2_settle*daily_data[t2],
                            'Overall Net':-t1_settle*daily_data[t1]-t2_settle*daily_data[t2],
                            'Position':0},
                            ignore_index=True)
                        
                        # add one success
                        active_trades[name][2] += 1
                        
                        cash += -t1_settle*daily_data[t1]-t2_settle*daily_data[t2]
                        
                else:
                    pass
                    
                
                # if pair diverges outside of 2*limit, attempt to close it
                if ((zscore < -8) or (zscore > 8)):
                                                                                    
                    value = ((sum(active_trades[name][0]['T1 Shares'])*daily_data[t1])+
                                 (sum(active_trades[name][0]['T2 Shares'])*daily_data[t2]))
                    
                    if value > (2*amount):
                    
                        active_trades[name][0] = active_trades[name][0].append({
                            'Date':day,
                            'Zscore':zscore,
                            'T1 Price':daily_data[t1],
                            'T2 Price':daily_data[t2],
                            'T1 Shares':0,
                            'T2 Shares':0,
                            'T1 Net':sum(active_trades[name][0]['T1 Shares'])*daily_data[t1],
                            'T2 Net':sum(active_trades[name][0]['T2 Shares'])*daily_data[t2],
                            'Overall Net':((sum(active_trades[name][0]['T1 Shares'])*daily_data[t1])+
                                           sum(active_trades[name][0]['T2 Shares'])*daily_data[t2]),
                            'Position':2},
                            ignore_index=True)
                        
                        cash += ((sum(active_trades[name][0]['T1 Shares'])*daily_data[t1])+
                                           sum(active_trades[name][0]['T2 Shares'])*daily_data[t2])
                        
                        
                        # trade is closed, so add it to trade log
                        trade_log[name] = active_trades[name]
                        
                        # now delete key from active list
                        del active_trades[name]
                        
                        # change position var
                        ind = [i for i, val in enumerate((fpairs['Tick1']==t1) & (fpairs['Tick2']==t2)) if val] 
                        
                        if not ind:
                            pass
                        else:
                            position[ind[0]] = 5
    
                
            # if position = 3, the pair isn't cointegrated so close it
            elif pos == 3:
                    
                value = ((sum(active_trades[name][0]['T1 Shares'])*daily_data[t1])+
                                 (sum(active_trades[name][0]['T2 Shares'])*daily_data[t2]))
                
                if value > (2*amount):
                
                    active_trades[name][0] = active_trades[name][0].append({
                        'Date':day,
                        'Zscore':zscore,
                        'T1 Price':daily_data[t1],
                        'T2 Price':daily_data[t2],
                        'T1 Shares':0,
                        'T2 Shares':0,
                        'T1 Net':sum(active_trades[name][0]['T1 Shares'])*daily_data[t1],
                        'T2 Net':sum(active_trades[name][0]['T2 Shares'])*daily_data[t2],
                        'Overall Net':((sum(active_trades[name][0]['T1 Shares'])*daily_data[t1])+
                                       sum(active_trades[name][0]['T2 Shares'])*daily_data[t2]),
                        'Position':2},
                        ignore_index=True)
                    
                    cash += ((sum(active_trades[name][0]['T1 Shares'])*daily_data[t1])+
                                       sum(active_trades[name][0]['T2 Shares'])*daily_data[t2])
                    
                    
                    # trade is closed, so add it to trade log
                    trade_log[name] = active_trades[name]
                    
                    # now delete key from active list
                    del active_trades[name]
                    
                    # change position var
                    ind = [i for i, val in enumerate((fpairs['Tick1']==t1) & (fpairs['Tick2']==t2)) if val] 
                    position[ind[0]] = 5
        except:
            print('Error with {}.'.format(name))
            del active_trades[name]
                
                
    return active_trades,trade_log,position,cash
    

    
def test_pairs(df,fpairs,day,active_trades,trade_log,cash):
    
    position = list(fpairs['position'])
    
    amount = 10 # specify how much to buy and sell
    
    act_day = dt.datetime.strptime(day,"%Y-%m-%d %H:%M:%S")
        
    try:
        daily_data = df.loc[act_day]
        
        # loop through all pairs
        if len(active_trades) > 0:
            active_trades,trade_log,position,cash = trade_active_pairs(fpairs,daily_data,position,
                                                                       active_trades,trade_log,day,amount,cash)
                
        # loop through inactive pairs to activate them
        active_trades,position,cash = check_inactive_pairs(fpairs,daily_data,position,active_trades,day,amount,cash)

    except:
        print('Error in reading date')
        time.sleep(0.5)
                            
    
    fpairs['position'] = position
    
    return fpairs,active_trades,trade_log,cash
            


# def main():   
    
cash = 10000
cash_tracker = []

total_invest = cash
    
# open the last 10 years worth of adj. close prices for r1000 stocks
with open('Pairs Trading Stock Data/stock_10y1d_price.pickle','rb') as f:
    df = pickle.load(f)

# define test range of days
testdays = pd.date_range('2011-01-01 00:00:00','2020-11-30 00:00:00',freq='B').strftime("%Y-%m-%d %H:%M:%S").tolist()

daily_actives = []
daily_trade_log = []

active_trades = {}
trade_log = {}

old_pairs = pd.DataFrame()
cc = 0
month = [0]
month_changes = 0
matches = [0]

# loop through each day in test range
for day in testdays:
    cc += 1
    cash_tracker.append(cash)
    
    print(day)
    print('Cash: ${}'.format(round(cash)))
    print('Active Trades: {}'.format(len(active_trades)))
    print('Closed Trades: {}'.format(len(trade_log)))
    print('')
    
    # define the specific day
    month.append(dt.datetime.strptime(day,"%Y-%m-%d %H:%M:%S").month)
    act_day = dt.datetime.strptime(day,"%Y-%m-%d %H:%M:%S")
    
    # if its a new month, recalculate stats
    if month[cc] != month[cc-1]:
        month_changes += 1
        
        if not month_changes == 1:
            profit = 0
            keys = list(trade_log)
            for k in keys:
                profit += sum(trade_log[k][0]['Overall Net'])
                
            act_profit = 0
            keys = list(active_trades)
            for k in keys:
                act_profit += sum(active_trades[k][0]['Overall Net'])+20
                
            print(act_day)    
            print('Net Profit: ${}'.format(round(profit+act_profit)))
            print('Net Return: {}%'.format(round((profit+act_profit)/10000*100/(month_changes/12),2)))
            print('')
        
        # extract pairs from the last 6 months, and create the test dataframe for the current month
        # create new pairs only once a month
        print('Testing for correlation...')
        pairs,train,test = extract_correlation_pairs(df,act_day,train_length=6,test_forward=1)
        
        # test each pair for cointegration
        # test new pairs only once a month
        print('Testing for cointegration...')
        est_time = len(pairs)/170
        finish_time = dt.datetime.fromtimestamp(time.time()+est_time).strftime('%H:%M:%S')
        print('Estimated Complete Time: {}'.format(finish_time))        

        tasks = list(zip(pairs['row'],pairs['col'],repeat(train)))
        pairs['pvals'] = [test_pair_cointegration(row) for row in tasks]
        
        pairs = pairs[pairs['pvals'] < 0.01]
        pairs.reset_index(drop=True,inplace=True)
        
        # calculate z stats for each pair, and only keep those with at least 1 successes in 6 months
        print('')
        print('Testing pairs...')
        fpairs = pair_cointegration_stats(pairs,df,act_day,train_length=3)
        
        fpairs
        
        # start each position at 2, meaning it has to return to the mean before we buy
        fpairs['position'] = [0]*len(fpairs)
        
        m = 0
        # if this isnt the first month change, transfer the old pairs over to fpairs
        if not month_changes == 1:
            position = [0]*len(old_pairs) 
            for ii in range(len(fpairs)):
                for jj in range(len(old_pairs)):
                    
                    if fpairs.loc[ii,'Tick1'] == old_pairs.loc[jj,'Tick1']:
                        if fpairs.loc[ii,'Tick2'] == old_pairs.loc[jj,'Tick2']:
                            
                            fpairs.loc[ii,'position'] = old_pairs.loc[jj,'position']
                            
                            m += 1
            
            
            matches.append(m)
                        
    
    # loop through the next month, testing the strategy
    fpairs,active_trades,trade_log,cash = test_pairs(df,fpairs,day,active_trades,trade_log,cash)

    
    old_pairs = fpairs
    
    
    daily_actives.append([day,active_trades.copy()])
    daily_trade_log.append([day,trade_log.copy()])
    
    
    
        
testdays_dt = [dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in testdays]
testdays = [dt.datetime.strftime(x,'%Y-%m-%d') for x in testdays_dt]


# plot the profit over time

cum_profit = []
ret_perc = []
old_profits = []
act_profits = []
act_length = []
for ii in range(len(testdays)):
    act = daily_actives[ii][1]
    act_length.append(len(act))
    
    profit = 0
    keys = list(act)
    for k in keys:
        profit += sum(act[k][0]['Overall Net'])+20
        
    if any(daily_trade_log[ii][1]):
        old_profit = 0
        keys = list(daily_trade_log[ii][1])
        for k in keys:
            old_profit += sum(daily_trade_log[ii][1][k][0]['Overall Net'])
        
    else:
        old_profit = 0
        
    old_profits.append(old_profit)
    act_profits.append(profit)
    
        
    cum_profit.append(profit+old_profit+cash_tracker[ii])
    ret_perc.append((profit+old_profit)/10000*100)
    

fig,ax = plt.subplots(2,1)
ax[0].plot(testdays,cum_profit,'b',linewidth=2,label='Net Worth')
ax[0].plot(testdays,cash_tracker,'g--',label='Cash')
ax[0].plot(testdays,old_profits,c='orange',label='Closed Profits')
ax[0].plot(testdays,act_profits,c='cyan',label='Active Profits')
ax[0].legend(loc='upper left')
ax[0].grid()
ax[0].xaxis.set_major_locator(plt.MaxNLocator(10))
ax[0].set_title('3 yr, $10 Actions')


ax[1].plot(testdays,act_length,label='# of Active Pairs')
ax[1].xaxis.set_major_locator(plt.MaxNLocator(10))
ax[1].grid()
ax[1].legend(loc='upper left')


keys = list(trade_log)
ret = []
ss = []
l = []
for k in keys:
    if trade_log[k][2]>4:
        ret.append(sum(trade_log[k][0]['Overall Net'])/20)
        ss.append(trade_log[k][2])
        l.append((dt.datetime.strptime(trade_log[k][0]['Date'].iloc[-1],'%Y-%m-%d %H:%M:%S') 
             - dt.datetime.strptime(trade_log[k][0]['Date'].iloc[0],'%Y-%m-%d %H:%M:%S')).days)
    
l = np.array(l)
ss = np.array(ss)
ret = np.array(ret)
    
plt.figure()
plt.plot(ss,(ret/(l/365)*100),'.')
plt.axhline(np.mean(ret/(l/365)*100),c='black',ls='dashed')
plt.grid()
plt.xlabel('Number of Successful Pair Trades')
plt.ylabel('Percent Return of Pair')
plt.title('Average Trade Return = 29% (2015-2018)')




with open('Results/active_10yr.pickle','wb') as f:
    pickle.dump(active_trades,f)
    
with open('Results/tradeLog_10yr.pickle','wb') as f:
    pickle.dump(trade_log,f)





