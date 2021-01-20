#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:05:17 2020

@author: brettdavis
"""

import pickle
import bz2
import _pickle as cPickle
import yahoo_fin.stock_info as yfin
import datetime as dt
import dateutil.relativedelta as rel_delta
from pandas.tseries.offsets import BDay
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint,adfuller
import statsmodels.api as sm
import signal
from contextlib import contextmanager
import json
import yfinance as yf
from pandas_datareader import data as web
yf.pdr_override()


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


def update_stock_history():
    ## Read the up-to-date stock adj. close values from yahoo_finance and save them
    
    # load the list of all possible tickers
    tickers = list(pd.read_csv('Sector Ticker Lists/AllTicks.csv')['Symbol'])
    
    # load in the un-updated dataframe
    with open('Saved Variables/stock_daily_close.pickle','rb') as f:
        df = pickle.load(f)
    
    # this var represents the last day the data was updated
    lastdate = df.index[-1]
    
    print('Loading monthly stock data for all tickers...')
    data = web.get_data_yahoo(tickers,start=lastdate)['Adj Close']
    
    # append the new data to the existing dataframe
    df = df.append(data[1:])
            
    with bz2.BZ2File('Saved Variables/stock_daily_close.pbz2','w') as f: 
        cPickle.dump(df,f)
        
def save_sp500(today,sp500comp,cash):
    try:
        start_sp500 = (today+rel_delta.relativedelta(months=-18))
        prices = yfin.get_data('^GSPC',start_date=start_sp500,end_date=today)['adjclose']
        dates = prices.index.strftime("%m-%d-%Y").values
        
        # add sp500 comparison data for dashboard chart
        if not sp500comp:
            sp500comp.append({'date': today.strftime("%m-%d-%Y"), 'spvalue': 10000.00,'myvalue': 10000.00})
        else:
            growth = (prices[-1] - prices[-2]) / prices[-2]
            
            sp500comp.append({
                'date': today.strftime("%m-%d-%Y"),
                'spvalue': round(sp500comp[-1]['spvalue'] + sp500comp[-1]['spvalue']*growth,2),
                'myvalue': round(cash['cash']+cash['holding']+cash['active'],2)
                    })
            
        
        
        sp500 = pd.DataFrame()
        sp500['dates'] = dates
        sp500['prices'] = prices.values.round(2)
        
        sp500.prices = sp500.prices.map('{:.2f}'.format)
        
        sp500.to_json('../stock_website/static/data/dashboard/sp500.json',orient='table',indent=4)
        
        return sp500comp
    
    except:
        print('Unable to save the S&P 500!')

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
    
def resave_sector_tick_list():
    data = pd.read_csv("Ticker Lists/AllTicks.csv")
    
    sect_list = data.Sector.unique()
    
    
    # pull out list of ticks for each sector
    sect_ticks = {}
    for sect in sect_list: 
        if str(sect) != 'nan':
            sect_ticks[sect] = data['Symbol'][data['Sector']==sect]
    
    with open('sect_tick_dict.pickle','wb') as f:
        pickle.dump(sect_ticks,f)
    
def extract_correlation_pairs(df,today,train_length=6,test_forward=1,resave_ticks = False): # train length in months

    if resave_ticks:
        resave_sector_tick_list()
    
    with open('Saved Variables/sect_tick_dict.pickle','rb') as f:
        sect_ticks = pickle.load(f)
        
    train = df.loc[(today+rel_delta.relativedelta(months=-train_length)):today]
    train = train[train>0]
    train = train[train<2500]
    train = train.dropna(axis=1)
    
    pairs = pd.DataFrame(columns=['row','col','corr'])
    for key in list(sect_ticks):
        tickList = sect_ticks[key]
        
        train_sect = train[train.columns.intersection(list(tickList.values))]
        
        # get pairs with 80% correlation or better
        cmat = train_sect.corr()
        
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
        
        sect_pairs = pd.DataFrame({'row':t1,'col':t2,'corr':vals})
        
        pairs = pairs.append(sect_pairs,ignore_index=True)
        
    print('There are {} correlated pairs.'.format(len(pairs)))
    print('')

    return pairs,train

def test_pair_cointegration(pairs,train):
    
    logtrain = np.log10(train)
    pvalue = []
    for count in range(len(pairs)):
        t1 = pairs['row'][count]
        t2 = pairs['col'][count]
        _,pval,_ = coint(logtrain[t1],logtrain[t2])
        
        pvalue.append(pval)
              
    return pvalue

def pair_cointegration_stats(pairs,df,today,train,train_length=3): #-------------------------------------------------------------------------------------------
    
    train = train.loc[(today+rel_delta.relativedelta(months=-train_length)):today] 
    
    fpairs = pd.DataFrame(columns=['Tick1','Tick2','beta','zavg','zstd','zmax','success'])
    
    for row in range(len(pairs)):
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
    fpairs = fpairs.nlargest(100,'success',keep='all')
    fpairs.reset_index(drop=True,inplace=True)
    
    print('')
    print('There are {} final fpairs.'.format(len(fpairs)))
    
    return fpairs


def check_inactive_pairs(fpairs,df_day,position,active_trades,today,amount,cash,openpairs): #------------------------------------------------------------
    for row in range(len(fpairs)):
        
        try:
            if position[row] == 0:
                t1 = fpairs['Tick1'][row]
                t2 = fpairs['Tick2'][row]
                name = t1+'_'+t2
                b = fpairs['beta'][row]
                
                mean = fpairs['zavg'][row]
                std = fpairs['zstd'][row]
                zval = np.log10(df_day[t2]) - b * np.log10(df_day[t1])
                
                zscore = (zval-mean)/std            
                
                data = [t1,t2,b,mean,std]
                
                if name in list(active_trades):
                    pass
                
                else:
                    if (-0.5 < zscore < 0.5):
                        if cash['cash'] >= amount*10:
                            
                            active_trades[name] = [pd.DataFrame({'Date':[today],'Zscore':[zscore],'T1 Price':[df_day[t1]],'T2 Price':[df_day[t2]],
                                                          'T1 Shares':[amount/df_day[t1]],'T2 Shares':[amount/df_day[t2]],
                                                          'T1 Net':[(-amount)],'T2 Net':[(-amount)],'Overall Net':[(-2*amount)],
                                                          'Position':[0]}),data,0]
                            position[row] = 1
                            cash['cash'] += -amount*2
                            
                            openpairs.append({'Tick1':t1,'T1price':df_day[t1],'T1shares':amount/df_day[t1],
                                                'Tick2':t2,'T2price':df_day[t2],'T2shares':amount/df_day[t2],
                                                'Zscore':zscore,'Profit':-2*amount})     
                        
        
        except:
            print('Inactive error with pair.')
            print(row)
            
                
    return active_trades,position,cash,openpairs


def trade_active_pairs(fpairs,df_day,position,active_trades,trade_log,today,amount,amount_initial,cash,closepairs,tradepairs): #----------------------------------------
    
    # This function loops through all active pairs and checks the Zscore and position to make buy/sell decisions
    
    # this limit is a zscore to indicate closing pair
    upper_limit = 8

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
            
            
            zval = np.log10(df_day[t2]) - b * np.log10(df_day[t1])
            
            zscore = (zval-mean)/std
            
# if pos = 0 --------------------------------------------------------------------------------------------------------------------------------------------  
            # check if I have a current position for pair
            if pos == 0:
                
                # z rises too high, buy $10 tick1 stock, sell $10 tick2
                if (zscore > 1) and (zscore < upper_limit):
                    
                    active_trades[name][0] = active_trades[name][0].append({
                        'Date':today,
                        'Zscore':zscore,
                        'T1 Price':df_day[t1],
                        'T2 Price':df_day[t2],
                        'T1 Shares':amount/df_day[t1],
                        'T2 Shares':-amount/df_day[t2],
                        'T1 Net':(-amount),
                        'T2 Net':(amount),
                        'Overall Net':0,
                        'Position':1},
                        ignore_index=True)
                    
                    tradepairs.append({
                        'Tick1':t1,
                        'T1price':df_day[t1],
                        'T1shares':amount/df_day[t1],
                        'Tick2':t2,
                        'T2price':df_day[t2],
                        'T2shares':-amount/df_day[t2],
                        'Zscore':zscore,
                        'Profit': 0})
                    
                    
                # ratio falls too low, sell $100 tick1, buy $100 tick2
                elif (zscore < -1) and (zscore > -upper_limit):
                    active_trades[name][0] = active_trades[name][0].append({
                        'Date':today,
                        'Zscore':zscore,
                        'T1 Price':df_day[t1],
                        'T2 Price':df_day[t2],
                        'T1 Shares':-amount/df_day[t1],
                        'T2 Shares':amount/df_day[t2],
                        'T1 Net':(amount),
                        'T2 Net':(-amount),
                        'Overall Net':0,
                        'Position':-1},
                        ignore_index=True)
                    
                    tradepairs.append({
                        'Tick1':t1,
                        'T1price':df_day[t1],
                        'T1shares':-amount/df_day[t1],
                        'Tick2':t2,
                        'T2price':df_day[t2],
                        'T2shares':amount/df_day[t2],
                        'Zscore':zscore,
                        'Profit': 0})
                    
                    
                # if neither happens, do nothing
                else:
                    pass
            
# if pos = 1 or -1 ----------------------------------------------------------------------------------------------------------------------------------
            # if I do have a current position
            elif abs(pos) == 1:
                    
                # is pos is 1, sell T1 and buy T2
                if pos == 1:
                    if (zscore < 0.2):
                        
                        # calculate total number of current shares owned for each stock
                        shares = [sum(active_trades[name][0]['T1 Shares']),sum(active_trades[name][0]['T2 Shares'])]
                        
                        # calculate the final number of shares in order to equal initial amount invested
                        shares_reset = [amount_initial/df_day[t1],amount_initial/df_day[t2]]
                        
                        # these vars represent the amount of shares to buy/sell to reset
                        # if need to buy, t_settle is positive
                        # if need to sell, t_settle is negative
                        t1_settle = shares_reset[0]-shares[0]
                        t2_settle = shares_reset[1]-shares[1]
                    
                        active_trades[name][0] = active_trades[name][0].append({
                            'Date':today,
                            'Zscore':zscore,
                            'T1 Price':df_day[t1],
                            'T2 Price':df_day[t2],
                            'T1 Shares':t1_settle,
                            'T2 Shares':t2_settle,
                            'T1 Net':-t1_settle*df_day[t1],
                            'T2 Net':-t2_settle*df_day[t2],
                            'Overall Net':-t1_settle*df_day[t1]-t2_settle*df_day[t2],
                            'Position':0},
                            ignore_index=True)
                        
                        tradepairs.append({
                            'Tick1':t1,
                            'T1price':df_day[t1],
                            'T1shares':t1_settle,
                            'Tick2':t2,
                            'T2price':df_day[t2],
                            'T2shares':t2_settle,
                            'Zscore':zscore,
                            'Profit': -t1_settle*df_day[t1]-t2_settle*df_day[t2]})
                        
                        # add one success
                        active_trades[name][2] += 1
                        
                        # add profit to cash
                        cash['cash'] += -t1_settle*df_day[t1]-t2_settle*df_day[t2]
                        
                        # add profit to profit
                        cash['profit'] += -t1_settle*df_day[t1]-t2_settle*df_day[t2]
                        
                        
                        
                # is pos is -1, buy T1 and sell T2
                elif pos == -1:
                    if (zscore > -0.2):
                        
                        # calculate total number of current shares owned for each stock
                        shares = [sum(active_trades[name][0]['T1 Shares']),sum(active_trades[name][0]['T2 Shares'])]
                        
                        # calculate the final number of shares in order to equal initial amount invested
                        shares_reset = [amount_initial/df_day[t1],amount_initial/df_day[t2]]
                        
                        # these vars represent the amount of shares to buy/sell to reset
                        # if need to buy, t_settle is positive
                        # if need to sell, t_settle is negative
                        t1_settle = shares_reset[0]-shares[0]
                        t2_settle = shares_reset[1]-shares[1]
                    
                        active_trades[name][0] = active_trades[name][0].append({
                            'Date':today,
                            'Zscore':zscore,
                            'T1 Price':df_day[t1],
                            'T2 Price':df_day[t2],
                            'T1 Shares':t1_settle,
                            'T2 Shares':t2_settle,
                            'T1 Net':-t1_settle*df_day[t1],
                            'T2 Net':-t2_settle*df_day[t2],
                            'Overall Net':-t1_settle*df_day[t1]-t2_settle*df_day[t2],
                            'Position':0},
                            ignore_index=True)
                        
                        tradepairs.append({
                            'Tick1':t1,
                            'T1price':df_day[t1],
                            'T1shares':t1_settle,
                            'Tick2':t2,
                            'T2price':df_day[t2],
                            'T2shares':t2_settle,
                            'Zscore':zscore,
                            'Profit': -t1_settle*df_day[t1]-t2_settle*df_day[t2]})
                        
                        # add one success
                        active_trades[name][2] += 1
                        
                        # add profit to cash
                        cash['cash'] += -t1_settle*df_day[t1]-t2_settle*df_day[t2]
                        
                        # add profit to profit
                        cash['profit'] += -t1_settle*df_day[t1]-t2_settle*df_day[t2]
                
                # if pos = 1 but meets no zscore criteria, just pass
                else:
                    pass
            
            # if nothing is done, pass (this isn't necessary)
            else:
                pass
            
# if zscore exceeds the upper limit, look to close it ----------------------------------------------------------------------------------------------------

            # if pair diverges outside of 2*limit, attempt to close it
            if ((zscore < -upper_limit) or (zscore > upper_limit)):
                                      
                # calculate the total value of the pair                                              
                value = ((sum(active_trades[name][0]['T1 Shares'])*df_day[t1])+
                             (sum(active_trades[name][0]['T2 Shares'])*df_day[t2]))
                
                # if the value is greater than the original amount invested, close pair by selling
                if value > (2*amount_initial):
                
                    active_trades[name][0] = active_trades[name][0].append({
                        'Date':today,
                        'Zscore':zscore,
                        'T1 Price':df_day[t1],
                        'T2 Price':df_day[t2],
                        'T1 Shares':-sum(active_trades[name][0]['T1 Shares']),
                        'T2 Shares':-sum(active_trades[name][0]['T2 Shares']),
                        'T1 Net':sum(active_trades[name][0]['T1 Shares'])*df_day[t1],
                        'T2 Net':sum(active_trades[name][0]['T2 Shares'])*df_day[t2],
                        'Overall Net':value,
                        'Position':2},
                        ignore_index=True)
                    
                    closepairs.append({
                        'Tick1':t1,
                        'T1price':df_day[t1],
                        'T1shares':-sum(active_trades[name][0]['T1 Shares']),
                        'Tick2':t2,
                        'T2price':df_day[t2],
                        'T2shares':-sum(active_trades[name][0]['T2 Shares']),
                        'Zscore':zscore,
                        'Profit': value})
                    
                    cash['cash'] += value
                    cash['profit'] += (value-amount_initial*2)
                    
                    # calculate stats for trade log            
                    ret = (((sum(active_trades[name][0]['Overall Net']))/amount_initial/2)*
                           (365/(today-(active_trades[name][0]['Date'][0])).days))*100
                    
                    
                    # pair is closed, so add it to trade log
                    trade_log.append({
                        'Tick1': t1,
                        'Tick2': t2,
                        'T1price': df_day[t1],
                        'T2price': df_day[t2],
                        'T1shares': sum(active_trades[name][0]['T1 Shares']),
                        'T2shares': sum(active_trades[name][0]['T2 Shares']),
                        'Zscore': zscore,
                        'Profit': sum(active_trades[name][0]['Overall Net']),
                        'Activity': (len(active_trades[name][0])-1)/2,
                        'Return': ret
                        }) 
                    
                    # now delete key from active list
                    del active_trades[name]
                    
                    # change fpairs position to inactive
                    ind = [i for i, val in enumerate((fpairs['Tick1']==t1) & (fpairs['Tick2']==t2)) if val] 
                    
                    if not ind:
                        pass
                    else:
                        position[ind[0]] = 0
            
            # if zscore exceeds limit but value is low, just wait for now
            else:
                pass
            
        # if there was a problem with the pair, print the name    
        except:
            print('Error with {}.'.format(name))
                
                
    return active_trades,trade_log,position,cash,closepairs,tradepairs
    

    
def test_pairs(df,fpairs,today,active_trades,trade_log,cash,sp500comp): #--------------------------------------------------------------------------------

    # define list of tickers to pull todays data for
    ticklist = list(fpairs['Tick1'])
    ticklist.extend(list(fpairs['Tick2']))
    
    #remove duplicates from the list
    ticklist = list(set(ticklist))
    
    # retrieve daily stock data
    print('Loading daily stock data...')
    df_day = web.get_data_yahoo(ticklist,start=today)['Adj Close'].iloc[0]            
    
    if len(df_day) > 10:

        position = list(fpairs['position'])
        
        amount = 10 # specify how much to buy and sell
        amount_initial = 15
        
        # today = dt.datetime.strptime(today,"%Y-%m-%d %H:%M:%S")
        
        openpairs = []
        closepairs = []
        tradepairs = []
        dashCard = {}
            
            
        # loop through all pairs
        if len(active_trades) > 0:
            active_trades,trade_log,position,cash,closepairs,tradepairs = trade_active_pairs(fpairs,df_day,position,
                                                                       active_trades,trade_log,today,amount,amount_initial,cash,closepairs,tradepairs)
                
        # loop through inactive pairs to activate them
        active_trades,position,cash,openpairs = check_inactive_pairs(fpairs,df_day,position,active_trades,today,amount_initial,cash,openpairs)
        
        fpairs['position'] = position
            
        
        
        # prepare variables for website ---------------------------------------
        dashCard['open'] = len(openpairs) 
        dashCard['trade'] = len(tradepairs)
        dashCard['close'] = len(closepairs)   
        dashCard['active'] = len(active_trades)
        dashCard['update'] = dt.datetime.now().strftime("%b %d" + " at " "%I:%M %p")            
        
        
        
        # create json for active trades
        active = []
        cash['active'] = 0
        cash['holding'] = 0
        for name in list(active_trades):
            
            try:
            
                # calculate necessary values
                t1 = active_trades[name][1][0]
                t2 = active_trades[name][1][1]
                
                b = active_trades[name][1][2]
                mean = active_trades[name][1][3]
                std = active_trades[name][1][4]
                zval = np.log10(df_day[t2]) - b * np.log10(df_day[t1])
                zscore = (zval-mean)/std
                
                prof = ((sum(active_trades[name][0]['T1 Shares'])*df_day[t1]) + 
                                (sum(active_trades[name][0]['T2 Shares'])*df_day[t2]) +
                                sum(active_trades[name][0]['Overall Net']))
                
                if today != (active_trades[name][0]['Date'][0]):
                    ret = (prof/amount_initial/2)*(365/(today-(active_trades[name][0]['Date'][0])).days)*100
                else:
                    ret = 0
                
                
                
                active.append({
                    'Tick1': t1,
                    'Tick2': t2,
                    'T1price': df_day[t1],
                    'T2price': df_day[t2],
                    'T1shares': sum(active_trades[name][0]['T1 Shares']),
                    'T2shares': sum(active_trades[name][0]['T2 Shares']),
                    'Zscore': zscore,
                    'Profit': prof,
                    'Activity': (len(active_trades[name][0])-1)/2,
                    'Return': ret,       
                    })
                
                # calculate total assets in active trades
                if abs(zscore)>8:
                    cash['holding'] += (sum(active_trades[name][0]['T1 Shares'])*df_day[t1])+(sum(active_trades[name][0]['T2 Shares'])*df_day[t2])
                else:
                    cash['active'] += (sum(active_trades[name][0]['T1 Shares'])*df_day[t1])+(sum(active_trades[name][0]['T2 Shares'])*df_day[t2])
                    
            except:
                pass
        
        
        # End active trade loop-----------------------------------------------
        
        # create json for action list ------------------------------------------
        catlist = openpairs+closepairs+tradepairs+tradepairs
        
        sharedict = {}
        for row in catlist:
            if row['Tick1'] in sharedict.keys():
                sharedict[row['Tick1']] += row['T1shares']
            else:
                sharedict[row['Tick1']] = row['T1shares']
                
            if row['Tick2'] in sharedict.keys():
                sharedict[row['Tick2']] += row['T2shares']
            else:
                sharedict[row['Tick2']] = row['T2shares']
        
        actionlist = []
        for key in list(sharedict):
            actionlist.append({'tick': key,'price':df_day[key],'shares': sharedict[key],'cost': (df_day[key]*sharedict[key])})
            
        # End of Action List loop ----------------------------------------------
        
        
        # save the SP500 data
        sp500comp = save_sp500(today,sp500comp,cash)
        
        with open('../stock_website/static/data/dashboard/sp500comp.json','w') as outfile:
            json.dump(sp500comp,outfile,indent=4)
        
        # Save all necessary variables for the website
        
        # var for daily action tables
        with open('../stock_website/static/data/openpairs.json','w') as outfile:
            json.dump(openpairs,outfile,indent=4)    
        with open('../stock_website/static/data/tradepairs.json','w') as outfile:
            json.dump(tradepairs,outfile,indent=4) 
        with open('../stock_website/static/data/closepairs.json','w') as outfile:
            json.dump(closepairs,outfile,indent=4)
            
        # var for action list
        with open('../stock_website/static/data/actionlist.json','w') as outfile:
            json.dump(actionlist,outfile,indent=4) 
            
        # var for dashboard cards
        with open('../stock_website/static/data/dashboard/dashCard.json','w') as outfile:
            json.dump(dashCard,outfile,indent=4)
        
        # var for active trades
        with open('../stock_website/static/data/active.json','w') as outfile:
            json.dump(active,outfile,indent=4)
            
        # var for trade log
        with open('../stock_website/static/data/tradelog.json','w') as outfile:
            json.dump(trade_log,outfile,indent=4)
            
        # var for asset pie chart
        with open('../stock_website/static/data/dashboard/pieChart.json','w') as outfile:
            json.dump(cash,outfile,indent=4)
        
    else:
        print('Market not open today.')
    
    return fpairs,active_trades,trade_log,cash,sp500comp
            


# def main():--------------------------------------------------------------------------------------------------------------------------------------------- 

## Alter this to test dates
testdays = pd.date_range('2016-01-01','2020-12-31',freq='B').tolist()
    
with bz2.BZ2File('Saved Variables/stock_daily_close.pbz2', 'rb') as f: 
    df = cPickle.load(f)
        
first_run = True 

if first_run:
    cash = {}
    cash['cash'] = 10000
    cash['profit'] = 0
    sp500comp = []
    
else:
    pass
          
for day in testdays:
    
    today = day
    
    # define the specific day
    yesterday = (today - BDay(1))
    
    yest_month = yesterday.month
    today_month = today.month
    
    # if its a new month or the first run, calculate new pair stats
    if (today_month != yest_month) or first_run:
        
        # save previous month of stock data
        update_stock_history()
        
        # if this isn't the first run, save last months pairs as old_pairs
        if not first_run:
            
            # although it says there's an error, the logic works with no issues
            old_pairs = fpairs
        
        # extract pairs from the last 6 months, and create the test dataframe for the current month
        # create new pairs only once a month
        print('Testing for correlation...')
        pairs,train = extract_correlation_pairs(df,today,train_length=6,test_forward=1)
        
        # test each pair for cointegration
        # test new pairs only once a month
        print('Testing for cointegration...')        
        pairs['pvals'] = test_pair_cointegration(pairs,train)
        
        pairs = pairs[pairs['pvals'] < 0.01]
        pairs.reset_index(drop=True,inplace=True)
        
        # calculate z stats for each pair, and only keep the top 200 in successes in n months
        print('')
        print('Calculating pair stats...')
        fpairs = pair_cointegration_stats(pairs,df,today,train,train_length=3)
        
        
        # start each position at 0, meaning it has to return to the mean before we buy
        fpairs['position'] = [0]*len(fpairs)
        
        # if this isnt the first month change, transfer the old pairs over to fpairs
        if not first_run:
            position = [0]*len(old_pairs) 
            for ii in range(len(fpairs)):
                for jj in range(len(old_pairs)):
                    
                    if fpairs.loc[ii,'Tick1'] == old_pairs.loc[jj,'Tick1']:
                        if fpairs.loc[ii,'Tick2'] == old_pairs.loc[jj,'Tick2']:
                            
                            fpairs.loc[ii,'position'] = old_pairs.loc[jj,'position']
                            
                            # signal that the old pair has been copied over
                            old_pairs.loc[jj,'position'] = -1
                            
            # any old pairs with positions that weren't copied over, ncopy them now
            copy_old_pairs = old_pairs[old_pairs['position']>0]
            
            # now append those old pairs to fpairs
            fpairs = fpairs.append(copy_old_pairs,ignore_index=True)
            
                            
            
                            
        
        ## End of recalculating stats for first of each month ----------------------
            
        
    # if this is the first run, create empty variables            
    if first_run:
        active_trades = {}
        trade_log = []
    
    
    # test the pairs each day and determine actions
    disp_today = today.strftime("%b %d, %Y")
    print('')
    print('Testing pairs on ' + disp_today +'...')
    fpairs,active_trades,trade_log,cash,sp500comp = test_pairs(df,fpairs,today,active_trades,trade_log,cash,sp500comp)
    
    
    # store all vars in one dict
    all_vars = {}
    all_vars['cash'] = cash
    all_vars['sp500comp'] = sp500comp
    all_vars['fpairs'] = fpairs
    all_vars['active_trades'] = active_trades
    all_vars['trade_log'] = trade_log
    
    with open('Saved Variables/all_vars.pickle','wb') as f:
        pickle.dump(all_vars,f)
    
    
    first_run = False
  

    

    
