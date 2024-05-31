
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting.test import GOOG
import talib 
import scipy
import math
import pandas_ta as ta
from dataclasses import dataclass
import mplfinance as mpf
import pivot

class signal:
    def __init__(self,data):
        self.data= data
    def ema(self):
        self.data["EMA50"] = ta.ema(self.data.Close, length=50)
    
    def rsi(self,timeperiod=14):
        # Using the talib library to calculate the values
        self.data['rsi'] = talib.RSI(self.data.Close, timeperiod=timeperiod)
        self.data = self.data.dropna()
        
    def stoch(self):
        # Normal implementation of stoch rsi incorrect in talib so correct code corresponding to values in github taken
        # https://gist.github.com/ultragtx/6831eb04dfe9e6ff50d0f334bdcb8460d
        period=14
        smoothK=3
        smoothD=3
        rsi = self.data.rsi

        # Calculate StochRSI 
        stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        self.data['K'] = 100*stochrsi.rolling(smoothK).mean()
        self.data['D'] = self.data.K.rolling(smoothD).mean()
    

    def atr(self,timeperiod =10):
        self.data['atr'] = talib.ATR(self.data.High,self.data.Low,self.data.Close, timeperiod = timeperiod)
    
    def pivots(self):
        pivots = pivot.Pivot(self.data)
        points = pivots.get_extremes()
        self.data['pivots'] = [0] * len(self.data.Close)
        index = points.index
        types = points.type
        for i in range(len(index)):
            self.data.pivots[index[i]] = types[index[i]]
            
        #print(index,types)
        
        rsi = self.data.rsi
        pivots = pivot.Pivot(rsi)
        points = pivots.get_extremes()
        self.data['pivots_rsi'] = [0] * len(self.data.Close)
        index = points.index
        types = points.type
        for i in range(len(index)):
            self.data.pivots_rsi[index[i]] = types[index[i]]
        #print(index,types)

        print(self.data.pivots)
        count = 0 
        for k in self.data.pivots:
            if k!=0:
                count+=1
        print(count)


        #print(self.data.pivots_rsi)

signals = signal(GOOG)
signals.rsi()
signals.pivots()




