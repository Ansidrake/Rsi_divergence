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

        self.pivot('high',self.data.High,'Pivot_high','gradient_high')
        self.pivot('high',self.data.rsi,'Pivot_high_rsi','gradient_high_rsi')
        self.pivot('low',self.data.Low,'Pivot_low','gradient_low')
        self.pivot('low',self.data.rsi,'Pivot_low_rsi','gradient_low_rsi')
        
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

summary = pd.DataFrame(columns=['Ticker', 'P/L', 'No.of trades', 'Return (%)'])

class Risk:
    def __init__(self, data, type):
        self.data = data
        # risk management strategy type
        self.type = type
    
    def set_stop_loss(self,entry_price,index,entry_type):
        if self.type == 'atr':
            if entry_type == 'long':
                stop_loss = entry_price - 2 * self.data.atr[index]
            elif entry_type == 'short':
                stop_loss = entry_price + 2 * self.data.atr[index]
        elif self.type == 'adjusting':
            if entry_type == 'long':
                sl = entry_price - (entry_price - self.data.Pivot_low[index])*1.5
                stop_loss = [sl]*4
            elif entry_type == 'short':
                sl = entry_price + (entry_price - self.data.Pivot_high[index])*1.5
                stop_loss = [sl]*4
        return stop_loss


    def update_stop_loss(self,entry_price,current_index,entry_type,last_stop):
        if self.type == 'atr':
            if entry_type == 'long':
                temp = self.data.Low[current_index] - 2 * self.data.atr[current_index]
                if temp > last_stop:
                    stop_loss = temp
                else:
                    stop_loss = last_stop
            elif entry_type == 'short':
                temp = self.data.Low[current_index] + 2 * self.data.atr[current_index]
                if temp < last_stop:
                    stop_loss = temp
                else:
                    stop_loss = last_stop         
        elif self.type == 'adjusting':
            stop_loss = last_stop
            
            if entry_type == 'long':
                treshold = entry_price - self.data.Pivot_low[current_index]
                best_case = [entry_price + i * treshold for i in range(1,5)]
                current_price = self.data.Close[current_index]
                # treshold for updating 
                check = treshold * 0.75
                if entry_price <= current_price <= best_case[0]:
                    if entry_price + check < current_price:
                        stop_loss = [entry_price]*4
                elif best_case[0] <= current_price <= best_case[1]:
                    if best_case[0] + check < current_price:
                        stop_loss = [entry_price,best_case[0],best_case[0],best_case[0]]
                elif best_case[1] <= current_price <= best_case[2]:
                    if best_case[1] + check < current_price:
                        stop_loss = [entry_price,best_case[0],best_case[1],best_case[1]]
                elif best_case[2] <= current_price <= best_case[3]:
                    if best_case[1] + check < current_price:
                        stop_loss = [entry_price,best_case[0],best_case[1],best_case[2]]
                
            elif entry_type == 'short':
                treshold = entry_price - self.data.Pivot_low[current_index]
                best_case = [entry_price - i * treshold for i in range(1,5)]
                current_price = self.data.Close[current_index]
                # treshold for updating 
                check = treshold * 0.75
                if entry_price >= current_price >= best_case[0]:
                    if entry_price - check > current_price:
                        stop_loss = [entry_price]*4
                elif best_case[0] >= current_price >= best_case[1]:
                    if best_case[0] - check > current_price:
                        stop_loss = [entry_price,best_case[0],best_case[0],best_case[0]]
                elif best_case[1] >= current_price >= best_case[2]:
                    if best_case[1] - check > current_price:
                        stop_loss = [entry_price,best_case[0],best_case[1],best_case[1]]
                elif best_case[2] >= current_price >= best_case[3]:
                    if best_case[1] - check > current_price:
                        stop_loss = [entry_price,best_case[0],best_case[1],best_case[2]]
        return stop_loss
            
    
    def set_take_profit(self,entry_price,index,entry_type):
        if self.type == 'atr':
            if entry_type == 'long':
                take_profit = entry_price + 2 * self.data.atr[index]
            elif entry_type == 'short':
                take_profit = entry_price - 2 * self.data.atr[index]
        elif self.type == 'adjusting':
            if entry_type == 'long':
                tp = entry_price + (entry_price - self.data.Pivot_low[index])
                take_profit = [tp]*4
            elif entry_type == 'short':
                sl = entry_price - (entry_price - self.data.Pivot_high[index])
                take_profit = [tp]*4
        return take_profit

    def update_take_profit(self,entry_price,current_index,entry_type,last_tp):
        take_profit = last_tp
        if self.type == 'atr':
           if entry_type == 'long':
               temp = self.data.High[current_index] + 2 * self.data.atr[current_index]
               if temp > last_tp:
                   take_profit = temp
           elif entry_type == 'short':
               temp = self.data.Low[current_index] - 2 * self.data.atr[current_index]
               if temp < last_tp:
                   take_profit = temp  
        
        elif self.type == 'adjusting':
            take_profit = last_tp
            
            if entry_type == 'long':
                treshold = entry_price - self.data.Pivot_low[current_index]
                best_case = [entry_price + i * treshold for i in range(1,5)]
                current_price = self.data.Close[current_index]
                # treshold for updating 
                check = treshold * 0.75
                if entry_price <= current_price <= best_case[0]:
                    if entry_price + check < current_price:
                        take_profit = [best_case[0],best_case[1],best_case[1],best_case[1]]
                elif best_case[0] <= current_price <= best_case[1]:
                    if best_case[0] + check < current_price:
                        take_profit = [best_case[0],best_case[1],best_case[2],best_case[2]]
                elif best_case[1] <= current_price <= best_case[2]:
                    if best_case[1] + check < current_price:
                        take_profit = [best_case[0],best_case[1],best_case[2],best_case[3]]
            
            elif entry_type == 'short':
                treshold = entry_price - self.data.Pivot_high[current_index]
                best_case = [entry_price - i * treshold for i in range(1,5)]
                current_price = self.data.Close[current_index]
                # treshold for updating 
                check = treshold * 0.75
                if entry_price >= current_price >= best_case[0]:
                    if entry_price - check > current_price:
                        take_profit = [best_case[0],best_case[1],best_case[1],best_case[1]]
                elif best_case[0] >= current_price >= best_case[1]:
                    if best_case[0] - check > current_price:
                        take_profit = [best_case[0],best_case[1],best_case[2],best_case[2]]
                elif best_case[1] >= current_price >= best_case[2]:
                    if best_case[1] - check > current_price:
                        take_profit = [best_case[0],best_case[1],best_case[2],best_case[3]]
        return take_profit

class strategy:
    def __init__(self,ticker,risk_strategy):
        self.start_time = time.time()
        stocks =  Stocks(ticker)
        self.stocks = stocks
        # uncomment the risk strategy you want to apply
        
        risk = Risk(self.stocks.data,risk_strategy)
        
        self.capital = 100000
        self.open_position = False
        self.buy_price = 0
        self.buy_qty = 0
        self.sell_price = 0
        self.sell_qty = 0
        self.tradetype = None
        self.trades = pd.DataFrame(columns=['Buy price', 'Sell price', 'Quantity','Trade type', 'PNL', 'Return (%)','Capital'])
        self.trades.loc[0] = [0,0,0,'None',0,0,10000]
        self.take_profit = None
        self.stop_loss = None
        
        self.number = 0
    
    def reinitialize(self):
            self.buy_price,self.buy_qty,self.sell_price,self.sell_qty = 0,0,0,0
            self.open_position = False
            self.tradetype = None
            self.take_profit = None
            self.stop_loss = None

    def long(self,index,qty):
        # Squarring off
        if self.open_position:
            self.buy_price = self.stocks.data.price[index]
            self.buy_qty = qty
            pnl = (self.sell_price - self.buy_price) * qty
            pnl_percent = (pnl/self.capital) * 100
            self.capital += pnl
            self.trades.loc[len(self.trades.index)] = [self.buy_price,self.sell_price,qty,self.tradetype,pnl,pnl_percent,self.capital]
            self.number += 1
            
        else:
            self.buy_price = self.stocks.data.price[index]
            if self.capital < self.buy_price:
                pass
            else:
                self.buy_qty = self.capital // self.buy_price
            self.open_position = True
            self.tradetype = 'long'
            self.number += 1
    

    def short(self,index,qty):
        # Squarring off
        if self.open_position:
            self.sell_price = self.stocks.data.price[index]
            self.sell_qty = qty
            pnl = (self.sell_price - self.buy_price) * qty
            pnl_percent = (pnl/self.capital) * 100
            self.capital += pnl
            self.trades.loc[len(self.trades.index)] = [self.buy_price,self.sell_price,qty,self.tradetype,pnl,pnl_percent,self.capital]
            # reinitializing the conditions
            self.buy_price,self.buy_qty,self.sell_price,self.sell_qty = 0,0,0,0
            self.open_position = False
            self.tradetype = None
            self.number += 1
        else:
            self.sell_price = self.stocks.data.price[index]
            if self.capital < self.buy_price:
                pass
            else:
                self.sell_qty = self.capital // self.sell_price
            self.open_position = True
            self.tradetype = 'short'
            self.number += 1

    def condition(self,index):
        pass

    def run_strategy(self):
        for i in range(len(self.stocks.data.Close)):
            if self.open_position:
                if self.tradetype == 'long':
                    self.risk.update_stop_loss(self.buy_price,i,self.tradetype,self.stop_loss)
                    self.risk.update_take_profit(self.buy_price,i,self.tradetype,self.take_profit)
                elif self.tradetype == 'short':
                    self.risk.update_stop_loss(self.sell_price,i,self.tradetype,self.stop_loss)
                    self.risk.update_take_profit(self.sell_price,i,self.tradetype,self.take_profit)
            self.condition(i)
        
        print(self.trades)
        summary.loc[len(summary)] = [self.stocks.ticker,round(self.capital-10000,2),self.number,round(((self.capital-10000)/10000)*100,2)]
        
        print(self.__repr__())

    def __repr__(self):
        self.end_time = time.time()
        return f"| Ticker ==> {self.stocks.ticker} | p/l ==>  {round(self.capital-10000,2)} ( {round(((self.capital-10000)/10000)*100,2)} % ) | time taken ==>  {round(self.end_time-self.start_time,2)} s |"

