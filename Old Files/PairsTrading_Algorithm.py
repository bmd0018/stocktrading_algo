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

def get_10y1d_data(reload_r1000=False):
    if reload_r1000:
        tickers = get_r1000_tickers()
    else:
        with open('r1000ticks.pickle','rb') as f:
            tickers = pickle.load(f)
      
    df_price = pd.DataFrame()             
    for tick in tickers:
        try:
            
            
            sdf = yf.get_data(tick,start_date='2010-01-01 00:00:00')
            df_price[tick] = sdf['adjclose']
                
            print('Finished: {}'.format(tick))
            
        except:
            print('No Content: {}'.format(tick))
            
    corrmatrix = df_price.corr()
    
    with open('Pairs Trading Stock Data/correlation_matrix_10y1d.pickle','wb') as f:
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


    
def extract_correlation_pairs(df,today,train_length=6,test_forward=1): # train length in months
    
    train = df.loc[(today+rel_delta.relativedelta(months=-train_length)):today]     
    test = df.loc[today:(today+rel_delta.relativedelta(months=test_forward))] 
    
    # get pairs with 80% correlation or better
    cmat = train.corr()
    pairs = pd.DataFrame(columns=['row','col','corr'])
    
    widgets = [pgb.FormatLabel('Extracting Pairs: '),pgb.Bar('*'),pgb.Timer(format= '  Timer:   %(elapsed)s'),'   ',pgb.ETA()] 
    bar = pgb.ProgressBar(max_value=len(cmat.index),widgets=widgets)
    
    rc = 0
    for row in cmat.index:
        rc += 1
        bar.update(rc)
        for col in cmat.columns:
            if (row!=col) and ((cmat.at[row,col]) >= 0.80):
                if (any(n in col for n in pairs['row'])) and (any(n in row for n in pairs['col'])):
                    pass
                else:
                    pairs = pairs.append({'row':row,'col':col,'corr':cmat.at[row,col]},ignore_index=True)
                       
    return pairs,train,test


def test_pair_cointegration(pairs,train,p_limit=0.001):
    
    coint_pairs = pd.DataFrame(columns=('Tick1','Tick2'))
    
    widgets = [pgb.FormatLabel('Testing for Cointegration: '),pgb.Bar('*'),pgb.Timer(format= '  Timer:   %(elapsed)s'),'   ',pgb.ETA()] 
    bar = pgb.ProgressBar(max_value=len(pairs),widgets=widgets)
    
    for row in range(len(pairs)):
        bar.update(row)
        
        logvals = pd.DataFrame(columns=('Tick1','Tick2'))
        
        t1 = pairs['row'][row]
        t2 = pairs['col'][row]
        
        logvals['Tick1'] = np.log10(train[t1])
        logvals['Tick2'] = np.log10(train[t2])
        
        logvals.dropna(inplace=True)
        
        _,pvalue,_ = coint(logvals['Tick1'],logvals['Tick2'])
        
        if pvalue < p_limit:
            coint_pairs = coint_pairs.append({'Tick1':t1,'Tick2':t2},ignore_index=True)
            # print('{}%'.format(round(row/len(pairs),3)))
            
    pairs = coint_pairs.reset_index(drop=True)
            
    return pairs

                
def pair_cointegration_stats(pairs,train):
    
    # widgets = [pgb.FormatLabel('Calculating Pair Stats: '),pgb.Bar('*'),pgb.Timer(format= '  Timer:   %(elapsed)s'),'   ',pgb.ETA()] 
    # bar = pgb.ProgressBar(max_value=len(pairs),widgets=widgets)
    
    fpairs = pd.DataFrame(columns=['Tick1','Tick2','beta','zavg','zstd','zmax','success'])
    for row in range(len(pairs)):
        # bar.update(row)
        logvals = pd.DataFrame(columns=('Tick1','Tick2'))
        
        t1 = pairs['Tick1'][row]
        t2 = pairs['Tick2'][row]
        
        logvals['Tick1'] = np.log10(train[t1])
        logvals['Tick2'] = np.log10(train[t2])
        
        logvals.dropna(inplace=True)
        
        if len(logvals) >= 50:
    
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
                    print('Error!')
                
            if success >= 1:
                fpairs = fpairs.append({'Tick1':t1,'Tick2':t2,'beta':B,'zavg':zmean,'zstd':zstd,
                                        'zmax':zmax,'success':success},ignore_index=True)
                
            
    fpairs.reset_index(drop=True,inplace=True)
    
    
    return fpairs


def recalc_daily_coint_stats(fpairs,yesterday,df):

    train = df.loc[(yesterday+rel_delta.relativedelta(months=-6)):yesterday]  

    B = []
    zmean = []
    zstd = []
    zmax = []
    
    for row in range(len(fpairs)):
        # bar.update(row)
        logvals = pd.DataFrame(columns=('Tick1','Tick2'))
        
        t1 = pairs['Tick1'][row]
        t2 = pairs['Tick2'][row]
        
        logvals['Tick1'] = np.log10(train[t1])
        logvals['Tick2'] = np.log10(train[t2])
        
        logvals.dropna(inplace=True)
    
        add1s = sm.add_constant(logvals)
        results = sm.OLS(add1s['Tick2'],add1s.drop(columns='Tick2')).fit()
        
        B.append((results.params['Tick1']))
        z = logvals['Tick2'] - B[row]*logvals['Tick1']
        
        zmean.append(z.mean())
        zstd.append(z.std())
        zmax.append(abs(z-z.mean()).max())
        
        
    # fpairs['beta'] = B
    fpairs['zavg'] = zmean
    fpairs['zstd'] = zstd
    fpairs['zmax'] = zmax
                
            
    # fpairs.reset_index(drop=True,inplace=True)
    
    
    return fpairs

    
def test_pairs(df,fpairs,testdays,daily_results):

    # fig,ax = plt.subplots(2,1)
    
    # loop through each day over testdays
    position = list(fpairs['position'])
    shares = list(fpairs['shares'])
    daycount = 0
    for day in (testdays)[1:]:
        
        daycount += 1
        act_day = dt.datetime.strptime(day,"%Y-%m-%d %H:%M:%S")
        yesterday = dt.datetime.strptime(testdays[daycount-1],"%Y-%m-%d %H:%M:%S")
        
        try:
            daily_data = df.loc[act_day]
            
            try:
                yest_data = df.loc[yesterday]
            except:
                date_bool = (df.index == act_day)
                ydate= df[np.roll(date_bool,-1)]
                yest_data = ydate.iloc[0]
            
            # fpairs = recalc_daily_coint_stats(fpairs,yesterday,df)
            
            day_profit = 0
            day_holds = 0
            opened = 0
            closed = 0
            success = 0
            fails = 0
            returned = 0
    
            axes = 0
            for row in range(len(fpairs)):
                t1 = fpairs['Tick1'][row]
                t2 = fpairs['Tick2'][row]
                b = fpairs['beta'][row]
                mean = fpairs['zavg'][row]
                std = fpairs['zstd'][row]
                limit = fpairs['zmax'][row]
                
                last_zval = np.log10(yest_data[t2]) - b * np.log10(yest_data[t1])
                zval = np.log10(daily_data[t2]) - b * np.log10(daily_data[t1])
                
                # ax[axes].plot(daycount,mean,'.k')
                # ax[axes].plot(daycount,zval,'.b')
                # ax[axes].plot(daycount,(mean+std),'.r')
                # ax[axes].plot(daycount,(mean-std),'.r')
                # ax[axes].plot(daycount,(mean+2*limit),'.y')
                # ax[axes].plot(daycount,(mean-2*limit),'.y')
                
                
                # check if I have a current position for pair
                if position[row] == 0:
                    
                    # z rises too high, buy $100 tick1 stock, sell $100 tick2
                    if (zval > mean+std) and (zval <= mean+limit):
                        shares[row] = [100/daily_data[t1], -100/daily_data[t2]]
                        position[row] = 1
                        opened += 1
                        
                        # ax[axes].axvline(daycount,color='orange')
                        
                    # ratio falls too low, buy $100 tick2 stock, sell $100 tick1
                    elif (zval < mean-std) and (zval >= mean-limit):
                        shares[row] = [-100/daily_data[t1], 100/daily_data[t2]]
                        position[row] = 1
                        opened += 1
                        
                        # ax[axes].axvline(daycount,color='orange')
                        
                    # if neither happens, do nothing
                    else:
                        pass
                
                # if I do have a current position
                elif position[row] == 1:
                    
                    # if ratio falls back to mean, close position
                    if (last_zval < mean < zval) or (zval < mean < last_zval):
                        day_profit += (daily_data[t1]*shares[row][0])+(daily_data[t2]*shares[row][1])
                        
                        shares[row] = [0,0]
                        position[row] = 0
                        success += 1
                        
                        closed += 1
                        
                        # ax[axes].axvline(daycount,color='cyan')
                    
                    # if pair diverges outside of 2*std, close it
                    elif (zval > mean+2*limit) or (zval < mean-2*limit):
                        day_profit += (daily_data[t1]*shares[row][0])+(daily_data[t2]*shares[row][1])
    
                        shares[row] = [0,0]
                        position[row] = 2
                        fails += 1
                                            
                        # ax[axes].axvline(daycount,color='red')
                    
                    else:
                        pass
                    
                # if position = 2, we're not considering the pair until it returns to normal
                elif position[row] == 2:
                    # if ratio returns to normal, we'll consider it again
                    if (last_zval < mean < zval) or (zval < mean < last_zval):
                        position[row] = 0
                        returned += 1
                        # ax[axes].axvline(daycount,color='green')
                    else:
                        pass
                    
                elif position[row] == 3:
                    day_profit += (daily_data[t1]*shares[row][0])+(daily_data[t2]*shares[row][1])
    
                    shares[row] = [0,0]
                    position[row] = 2
                    fails += 1
                    
                    # ax[axes].axvline(daycount,color='red')
                
                else:
                    pass
                
                axes+=1
            
            # ax[2].plot(rel_day,profit,'.k')
            # ax[2].grid()
            
            
            current_holds = 0
            for ii in range(len(shares)):
                current_holds += ((daily_data[t1]*shares[ii][0])+(daily_data[t2]*shares[ii][1]))*-1
                
            current_profit = daily_results['Cumulative Profits'].iloc[-1] + day_profit
                
            daily_results = daily_results.append(pd.DataFrame([[day,current_profit,current_holds,day_profit,day_holds,opened,closed,fails,returned]],
                                                 columns=['Date','Cumulative Profits','Cumalitive Holds','Daily Profit',
                                          'Daily Holds','Opened','Closed','Failed','Returned']))
            
            print('')
            print(day)
            print('Total Profit: ${}'.format(round(current_profit)))
            print('Current Holds: ${}'.format(round(current_holds)))
            print('')
            
        except:
            print('')
            print('Error in reading date')
                            
       
    # fig,ax = plt.subplots(2,1)
    # ax[0].plot(days,daily_net,'k')
    # ax[0].plot(days,success_chart,'g')
    # ax[0].plot(days,fail_chart,'r')
    # ax[0].grid()
    
    # ax[1].plot(days,holds)
    # ax[1].grid()
    
    fpairs['position'] = position
    fpairs['shares'] = shares

    return fpairs,daily_results
            


# def main():   
    
# open the last 10 years worth of adj. close prices for r1000 stocks
with open('Pairs Trading Stock Data/stock_10y1d_price.pickle','rb') as f:
    df = pickle.load(f)

# define what months to test (days will be generated)
months = pd.date_range('2019-05-01 00:00:00','2020-10-01 00:00:00', 
          freq='MS').strftime("%Y-%m-%d %H:%M:%S").tolist()

# create an empty df to store the daily data
daily_results = pd.DataFrame([[0,0,0,0,0,0,0,0,0]],columns=['Date','Cumulative Profits','Cumalitive Holds','Daily Profit',
                                      'Daily Holds','Opened','Closed','Failed','Returned'])

old_pairs = pd.DataFrame()
matches = []
cc = 0
# loop through the first day of each month
for month in months:
    cc += 1
    
    # define the specific day
    today = dt.datetime.strptime(month,"%Y-%m-%d %H:%M:%S").date()
    
    # extract pairs from the last 3 months, and create the test dataframe for the current month
    # create new pairs only once a month
    pairs,train,test = extract_correlation_pairs(df,today,train_length=6,test_forward=1)
    
    # test each pair for cointegration
    # test new pairs only once a month
    pairs = test_pair_cointegration(pairs,train,p_limit=0.001)
    
    # calculate z stats for each pair, and only keep those with at least 1 successes in 6 months
    fpairs = pair_cointegration_stats(pairs,train)
    
    # for each pair, no position has been made
    fpairs['shares'] = [[0,0]]*len(fpairs)
    
    # start each position at 2, meaning it has to return to the mean before we buy
    fpairs['position'] = [2]*len(fpairs)
    
    m = 0
    if not cc == 1:
        position = [0]*len(old_pairs) 
        for ii in range(len(fpairs)):
            for jj in range(len(old_pairs)):
                
                if fpairs['Tick1'][ii] == old_pairs['Tick1'][jj]:
                    if fpairs['Tick2'][ii] == old_pairs['Tick2'][jj]:
                        
                        fpairs['shares'][ii] = old_pairs['shares'][jj]
                        fpairs['position'][ii] = old_pairs['position'][jj]
                        
                        position[jj] = -1
                        m += 1
        
        old_pairs['position'] = position
        old_pairs = old_pairs[old_pairs['position']!=-1]
        old_pairs = old_pairs[old_pairs['position']!=0]
        old_pairs = old_pairs[old_pairs['position']!=2]
        old_pairs['position'] = 3
        
        fpairs = fpairs.append(old_pairs)
        
        matches.append(m)
                        
                        
     
    # the date one month from now
    end = (today+rel_delta.relativedelta(months=1))
    
    # list of all business days between today and next month
    testdays = pd.date_range(today,end,freq='B').strftime("%Y-%m-%d %H:%M:%S").tolist()
    
    # loop through the next month, testing the strategy
    fpairs, daily_results = test_pairs(df,fpairs,testdays,daily_results)
    
    old_pairs = fpairs
    

daily_results.reset_index(drop=True,inplace=True)
daily_results = daily_results.drop(0)
daily_results['Date'] = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M:%S').date() for d in daily_results['Date']]    
daily_results.reset_index(drop=True,inplace=True)

spacing = 15
fig,ax = plt.subplots(3,1)
ax[0].plot(daily_results['Date'],daily_results['Cumulative Profits'])
for label in ax[0].xaxis.get_ticklabels()[::spacing]:
    label.set_visible(False) 
ax[0].grid()

data = [list(daily_results.iloc[0,5:])]
for ii in range(len(daily_results))[1:]:
    data2 = np.array(daily_results.iloc[ii,5:])
    data.append(list(data[ii-1] + data2))
    

ax[1].plot(daily_results['Date'],[data[jj][0] for jj in range(len(data))],'g',label='Opened')
ax[1].plot(daily_results['Date'],[data[jj][1] for jj in range(len(data))],'b',label='Closed')
ax[1].plot(daily_results['Date'],[data[jj][2] for jj in range(len(data))],'r',label='Failed')
ax[1].plot(daily_results['Date'],[data[jj][3] for jj in range(len(data))],'y',label='Returned')
for label in ax[1].xaxis.get_ticklabels()[::spacing]:
    label.set_visible(False)       
ax[1].legend()
ax[1].grid()


ax[2].plot(daily_results['Date'],daily_results['Cumalitive Holds'])
ax[2].grid()









