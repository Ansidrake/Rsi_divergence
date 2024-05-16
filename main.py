from pyclbr import Class
import numpy as np
import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm
import talib 
import warnings
warnings.filterwarnings('ignore')



class Stocks:
    
    def __init__(self,ticker,interval):
        # setting self.ticker to ticker string for further uses ahead 
        self.ticker = ticker
        self.interval = interval

        #self.fetch_data()
        # load data from cache if exists else call historical data and calculate all nessecary values
        try:
            self.data = self.load_data()
            
        except:
            print('not using cache')
            self.fetch_data()
            self.data.to_csv(f"rsi_divergence/cache/{self.ticker}_{self.interval}_data.csv")  
        
        print(self.data)

    def load_data(self):
        filename = f"rsi_divergence/cache/{self.ticker}_{self.interval}_data.csv"  

        return pd.read_csv(filename, index_col=0)

    def fetch_data(self):
        
        df_list = []
        data = yf.download(self.ticker, group_by="Ticker", period='2d',interval=self.interval,progress=False)
        df_list.append(data)
        df = pd.concat(df_list)
        
        # saving the downloaded data to the class
        
        self.data = df
        
        # execution price taken as next day's open
        self.data['price'] = self.data.Open.shift(-1)
        self.data = self.data.dropna()

        # calculating nessecary values using the below functions
        self.rsi()
        self.stoch()
        self.data = self.data.dropna()

        self.pivot('high',self.data.High,'Pivothigh','gradient_high')
        self.pivot('high',self.data.rsi,'Pivothigh_rsi','gradient_high_rsi')
        self.pivot('low',self.data.Low,'Pivotlow','gradient_low')
        self.pivot('low',self.data.rsi,'Pivotlow_rsi','gradient_low_rsi')
        
        self.atr()


        #self.data = self.data.dropna()
        

    def rsi(self):
        # Using the talib library to calculate the values
        self.data['rsi'] = talib.RSI(self.data.Close, timeperiod=14)
        self.data = self.data.dropna()
        #print(self.data)
    
    def stoch(self):
        # Normal implementation of stoch rsi incorrect in talib so correct code corresponding to values in github taken
        # https://gist.github.com/ultragtx/6831eb04dfe9e6ff50d0f334bdcb847d
        period=14
        smoothK=3
        smoothD=3
        rsi = self.data.rsi

        # Calculate StochRSI 
        stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        self.data['K'] = 100*stochrsi.rolling(smoothK).mean()
        self.data['D'] = self.data.K.rolling(smoothD).mean()
    

    def atr(self):
        self.data['atr'] = talib.ATR(self.data.High,self.data.Low,self.data.Close, timeperiod = 3)
    
    def pivot(self, type, source, name, name_gradient):
        pivot = [None] * len(source)
        if type == 'high':
            for i in range(5, len(source)-5):
                high = source[i-5:i+6].max()
                pivot[i] = high
            for i in range(5):
                pivot[i-5] = source[i-5:].max()
                pivot[i] = source[:i+6].max()
            self.data[name] = pivot
            self.gradient(name,name_gradient)
        elif type == 'low':
            for i in range(5, len(source)-5):
                low = source[i-5:i+6].min()
                pivot[i] = low
            for i in range(5):
                pivot[i-5] = source[i-5:].min()
                pivot[i] = source[:i+6].min()
            self.data[name] = pivot
            self.gradient(name,name_gradient)
        
    def gradient(self,name,name_gradient):
        gradient = [0]*len(self.data[name])
        temp = self.data[name].iloc[0]
        for i in range(len(self.data[name])):
            if self.data[name].iloc[i] != temp:
                gradient[i] = 100*(self.data[name].iloc[i] - temp)/temp
                temp = self.data[name].iloc[i]
            else:
                gradient[i] = gradient[i-1] 
        self.data[name_gradient] = gradient
    


Tsla = Stocks('TSLA','5m')
