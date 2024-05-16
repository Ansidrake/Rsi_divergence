import numpy as np
import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm
import talib 



class Stocks:
    
    def __init__(self,ticker,interval):
        # setting self.ticker to ticker string for further uses ahead 
        self.ticker = ticker
        self.interval = interval


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
        #self.pivot()
        self.stoch()
        self.pivot_high()
        self.pivot_low()
        self.pivot_high_rsi()
        self.pivot_low_rsi()

        self.data = self.data.dropna()
        

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
    

    def pivot_high(self):
        pivot_high = [None] * len(self.data.High)
        for i in range(5, len(self.data.High)-5):
            high = self.data.High[i-5:i+6].max()
            pivot_high[i] = high
        for i in range(5):
            pivot_high[i] = self.data.High[5-i:].max()
        self.data['Pivot_high'] = pivot_high
        self.gradient_high()
        
    def gradient_high(self):
        gradient = [0]*len(self.data.High)
        temp = self.data.Pivot_high[0]
        for i in range(len(self.data.Pivot_high)):
            if self.data.Pivot_high[i] != temp:
                gradient[i] = 100*(self.data.Pivot_high[i] - temp)/temp
                temp = self.data.Pivot_high[i]
            else:
                gradient[i] = gradient[i-1] 
        self.data['gradient_high'] = gradient
    
    def pivot_high_rsi(self):
        pivot_high = [None] * len(self.data.High)
        for i in range(5, len(self.data.High)-5):
            high = self.data.rsi[i-5:i+6].max()
            pivot_high[i] = high
        for i in range(5):
            pivot_high[i] = self.data.rsi[5-i:].max()
        self.data['Pivot_high_rsi'] = pivot_high
        self.gradient_high_rsi()
        
    def gradient_high_rsi(self):
        gradient = [0]*len(self.data.High)
        temp = self.data.Pivot_high_rsi[0]
        for i in range(len(self.data.Pivot_high)):
            if self.data.Pivot_high_rsi[i] != temp:
                gradient[i] = 100*(self.data.Pivot_high_rsi[i] - temp)/temp
                temp = self.data.Pivot_high_rsi[i]
            else:
                gradient[i] = gradient[i-1] 
        self.data['gradient_high_rsi'] = gradient
    
    def pivot_low(self):
        pivot_high = [None] * len(self.data.High)
        for i in range(5, len(self.data.High)-5):
            high = self.data.Low[i-5:i+6].min()
            pivot_high[i] = high
        for i in range(5):
            pivot_high[i] = self.data.Low[5-i:].min()
        self.data['Pivot_low'] = pivot_high
        self.gradient_low()
        
    def gradient_low(self):
        gradient = [0]*len(self.data.High)
        temp = self.data.Pivot_low[0]
        for i in range(len(self.data.Pivot_low)):
            if self.data.Pivot_low[i] != temp:
                gradient[i] = 100*(self.data.Pivot_low[i] - temp)/temp
                temp = self.data.Pivot_low[i]
            else:
                gradient[i] = gradient[i-1] 
        self.data['gradient_low'] = gradient

    def pivot_low_rsi(self):
        pivot_high = [None] * len(self.data.High)
        for i in range(5, len(self.data.High)-5):
            high = self.data.rsi[i-5:i+6].min()
            pivot_high[i] = high
        for i in range(5):
            pivot_high[i] = self.data.Low[5-i:].min()
        self.data['Pivot_low_rsi'] = pivot_high
        self.gradient_low_rsi()
        
    def gradient_low_rsi(self):
        gradient = [0]*len(self.data.High)
        temp = self.data.Pivot_low_rsi[0]
        for i in range(len(self.data.Pivot_low_rsi)):
            if self.data.Pivot_low_rsi[i] != temp:
                gradient[i] = 100*(self.data.Pivot_low_rsi[i] - temp)/temp
                temp = self.data.Pivot_low_rsi[i]
            else:
                gradient[i] = gradient[i-1] 
        self.data['gradient_low_rsi'] = gradient

Tsla = Stocks('RELIANCE.NS','5m')
        
