import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm
import talib 

class Stocks:
    
    def __init__(self,ticker):
        # setting self.ticker to ticker string for further uses ahead 
        self.ticker = ticker

        self.fetch_data()

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

    def fetch_data(self):

        df_list = []
        data = yf.download(self.ticker, group_by="Ticker", period='2d',interval='5m',progress=False)
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
        self.data['k'],self.data['d'] = talib.STOCH(self.data.rsi,self.data.rsi, self.data.rsi, 14)
    
    def calc_pivot(self, index, prd):
        # Ensure index and period are within bounds
        if index < prd or index + prd >= len(self.data):
            return None

        # Extract the window for calculation
        window = self.data.High[index - prd : index + prd + 1].values
        max_value = max(window)
        high_max = max(window[:prd] + window[prd + 1:])

        # Check pivot condition
        if max_value == window[prd] and window[prd] > high_max:
            return window[prd]
        return None

    def pivot(self):
        pivot_high = [None] * len(self.data.High)
        for i in range(len(self.data.High)):
            pivot_high[i] = self.calc_pivot(i, 5)
        self.data['Pivot_high'] = pivot_high
        print(pivot_high)


        
    def pivot(self):
        len_right, len_left = 5, 5
        #self.data.insert(7,"Pivot_high",self.pivot(),True)
        #self.data['pivot_rsi_high'] = self.data.rsi.rolling(window=len_left, min_periods=1).max().shift(len_right)
        #self.data['pivot_low'] = self.data.Low.rolling(window=len_left, min_periods=1).min().shift(len_right)
        #self.data['pivot_rsi_low' ] = self.data.rsi.rolling(window=len_left, min_periods=1).min().shift(len_right)


Tsla = Stocks('RELIANCE.NS')
        
