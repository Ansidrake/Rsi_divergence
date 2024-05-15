from numpy import short
import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm
import talib 
import ta


class Stocks:
    
    def __init__(self,ticker):
        # setting self.ticker to ticker string for further uses ahead 
        self.ticker = ticker

        self.fetch_data('short')

        # load data from cache if exists else call historical data and calculate all nessecary values
        #try:
        #    self.data = self.load_data()
        #except:
        #    print('not using cache')
        #    self.fetch_data()
        #    self.data.to_csv(f"rsi_divergence/cache/{self.ticker}_data.csv")  
        
        print(self.data)

    def load_data(self):
        filename = f"rsi_divergence/cache/{self.ticker}_data.csv"  

        return pd.read_csv(filename, index_col=0)

    def fetch_data(self,timeframe):
        
        if timeframe =='long':
            interval = '30m'
        else:
            interval = '5m'

        df_list = []
        data = yf.download(self.ticker, group_by="Ticker", period='2d',interval=interval,progress=False)
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

        self.data = self.data.dropna()
        print(self.data)

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

    


Tsla = Stocks('RELIANCE.NS')
        
