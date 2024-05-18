import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm
import talib 
import pandas_ta as ta

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
            datetime = pd.read_csv(f"rsi_divergence/cache/{self.ticker}_{self.interval}_data.csv")['Datetime']
            date,time = [],[]
            for dt in datetime:
                dt = str(dt)
                date.append(str(dt[:6]))
                time.append(str(dt[6:]))
            self.date,self.time = date,time
        
        except:
            print('not using cache')
            self.fetch_data()
            location = f"rsi_divergence/cache/{self.ticker}_{self.interval}_data.csv"
            self.data.to_csv(location,date_format='%Y%m%d%H%M%S')
            self.data.to_csv(location,date_format='%Y%m%d%H%M%S')
            self.pivot('high',self.data.High,'Pivot_high','gradient_high','pivot_index_high_prev','pivot_index_high_next')
            self.pivot('high',self.data.rsi,'Pivot_high_rsi','gradient_high_rsi','pivot_index_high_rsi_prev','pivot_index_high_rsi_next')
            self.pivot('low',self.data.Low,'Pivot_low','gradient_low','pivot_index_low_prev','pivot_index_low_next')
            self.pivot('low',self.data.rsi,'Pivot_low_rsi','gradient_low_rsi','pivot_index_low_rsi_prev','pivot_index_low_rsi_next')
            self.regular_bullish_divergence()
            self.hidden_bullish_divergence()
            self.regular_bearish_divergence()
            self.hidden_bearish_divergence()
            self.data.to_csv(location,date_format='%Y%m%d%H%M%S')
            datetime = pd.read_csv(location)['Datetime']
            date,time = [],[]
            for dt in datetime:
                dt = str(dt)
                date.append(str(dt[:6]))
                time.append(str(dt[6:]))
            self.date,self.time = date,time
        
    def load_data(self):
        filename = f"rsi_divergence/cache/{self.ticker}_{self.interval}_data.csv"
        return pd.read_csv(filename)

    def fetch_data(self):
        
        df_list = []
        data = yf.download(self.ticker, group_by="Ticker", period='60d',interval=self.interval,progress=False)
        df_list.append(data)
        df = pd.concat(df_list)
        
        # saving the downloaded data to the class
        
        self.data = df
        
        # execution price taken as next day's open
        self.data['price'] = self.data.Open.shift(-1)
        self.ema()
        self.data = self.data.dropna()

        # calculating nessecary values using the below functions
        self.rsi()
        self.stoch()
        #self.data = self.data.dropna()

        self.atr()

        self.data = self.data.dropna()
    
    def ema(self):
        self.data["EMA50"] = ta.ema(self.data.Close, length=50)
    
    def rsi(self):
        # Using the talib library to calculate the values
        self.data['rsi'] = talib.RSI(self.data.Close, timeperiod=14)
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
    

    def atr(self):
        self.data['atr'] = talib.ATR(self.data.High,self.data.Low,self.data.Close, timeperiod = 10)
    
    def pivot(self, type, source, name, name_gradient, pivot_index_name_prev,pivot_index_name_next):
        pivot = [None] * len(source)
        if type == 'high':
            for i in range(5, len(source) - 5):
                if source[i] == max(source[i - 5:i + 6]):
                    pivot[i] = source[i]
            for i in range(5):
                if source[i] == max(source[:i + 6]):
                    pivot[i] = source[i]
                if source[-i-1] == max(source[-(i + 6):]):
                    pivot[-i-1] = source[-(i + 6)]

        elif type == 'low':
            for i in range(5, len(source) - 5):
                if source[i] == min(source[i - 5:i + 6]):
                    pivot[i] = source[i]
            for i in range(5):
                if source[i] == min(source[:i + 6]):
                    pivot[i] = source[i]
                if source[-i-1] == min(source[-(i + 6):]):
                    pivot[-i-1] = source[-(i + 6)]
        self.data[pivot_index_name_prev] = [0] * len(pivot)
        self.data[pivot_index_name_next] = [0] * len(pivot)
    
        for i in range(len(pivot)):
            if pivot[i] is None and i>0:
                self.data[pivot_index_name_prev].iloc[i] = self.data[pivot_index_name_prev].iloc[i-1]
            else:
                self.data[pivot_index_name_prev].iloc[i],self.data[pivot_index_name_next].iloc[i] = i,i
        for i in range(len(pivot)-1,-1,-1):
            flag = False
            if pivot[i] is not None:
                if flag and i < len(pivot)-1:
                    self.data[pivot_index_name_prev].iloc[i],self.data[pivot_index_name_next].iloc[i] = i,self.data[pivot_index_name_next].iloc[i+1]
                else:
                    self.data[pivot_index_name_prev].iloc[i],self.data[pivot_index_name_next].iloc[i] = i,i
                    flag = True
            elif i< len(pivot)-1:
                self.data[pivot_index_name_next].iloc[i] = self.data[pivot_index_name_next].iloc[i+1]


        self.data[name] = pivot
        self.gradient(name, name_gradient,pivot_index_name_prev,pivot_index_name_next)

    def gradient(self,name,name_gradient,pivot_index_name_prev,pivot_index_name_next):
        gradient = [0]*len(self.data[name])
        for i in range(len(self.data[name])):
            last,n = self.data[pivot_index_name_prev].iloc[i],self.data[pivot_index_name_next].iloc[i]
            gradient[i] = (self.data[name].iloc[n]-self.data[name].iloc[last])/self.data[name].iloc[last]
        self.data[name_gradient] = gradient
    
    def regular_bullish_divergence(self):
        # https://www.babypips.com/learn/forex/regular-divergence
        signal = [0] * len(self.data.Close)
        for index in range(len(self.data.Close)):
            for i in range(5):
                # to make the signal stronger or weaker we could also confirm that the condition was reverse and this is a potiential sign of reversal
                #if self.data.gradient_low_rsi[self.data.pivot_index_low_rsi_prev[index-i]] < 0 and self.data.gradient_low[self.data.pivot_index_low_prev[index-i]] < 0:
                if self.data.gradient_low_rsi[index-i] > 0 and self.data.gradient_low[index-i] < 0:
                    signal[index] = 1
        self.data['regular_bullish_divergence'] = signal
        
    
    def hidden_bullish_divergence(self):
        signal = [0] * len(self.data.Close)
        for index in range(len(self.data.Close)):
            for i in range(1):
                #if self.data.gradient_low_rsi[self.data.pivot_index_low_rsi_prev[index-i]] < 0 and self.data.gradient_low[self.data.pivot_index_low_prev[index-i]] < 0:
                if self.data.gradient_low_rsi[index] < 0 and self.data.gradient_low[index] > 0:
                    signal[index] = 1
        self.data['hidden_bullish_divergence'] = signal
    
    def regular_bearish_divergence(self):
        signal = [0] * len(self.data.Close)
        for index in range(len(self.data.Close)):
            for i in range(5):
                #if self.data.gradient_high_rsi[self.data.pivot_index_high_rsi_prev[index-i]] > 0 and self.data.gradient_high[self.data.pivot_index_high_prev[index-i]] > 0:
                if self.data.gradient_high_rsi[index-i] < 0 and self.data.gradient_high[index-i] > 0:
                    signal[index] = 1
        self.data['regular_bearish_divergence'] = signal
    
    def hidden_bearish_divergence(self):
        signal = [0] * len(self.data.Close)
        for index in range(len(self.data.Close)):
            for i in range(1):
                #if self.data.gradient_high_rsi[self.data.pivot_index_high_rsi_prev[index-i]] > 0 and self.data.gradient_high[self.data.pivot_index_high_prev[index-i]] > 0:
                if self.data.gradient_high_rsi[index] > 0 and self.data.gradient_high[index] < 0:
                    signal[index] = 1
        self.data['hidden_bearish_divergence'] = signal

    


#tsla = Stocks('TSLA','5m')
#for i in range(len(tsla.data.Date)):
#    print(tsla.data.Date[i])

summary = pd.DataFrame(columns=['Ticker', 'P/L', 'No.of trades', 'Return (%)','Win %','Avg_win_value','Avg_loss_value'])

class Risk:
    def __init__(self, data, type):
        self.data = data
        # risk management strategy type
        self.type = type
    
    def set_stop_loss(self,entry_price,index,entry_type,initial):
        stop_loss = initial
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
        stop_loss = last_stop
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
            
    
    def set_take_profit(self,entry_price,index,entry_type,initial):
        take_profit = initial
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
                tp = entry_price - (entry_price - self.data.Pivot_high[index])
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
    def __init__(self,ticker,risk_strategy,strategy,timeframe):
        self.start_time = time.time()
        self.timeframe = timeframe
        stocks =  Stocks(ticker,timeframe)
        self.stocks = stocks
        # uncomment the risk strategy you want to apply
        
        self.risk = Risk(self.stocks.data,risk_strategy)
        self.risk_strategy = risk_strategy
        self.strategy = strategy
        if strategy == 'multi':
            self.longer = Stocks(ticker, '30m')
            datetime = [int(str(date) + str(time)) for date, time in zip(self.stocks.date, self.stocks.time)]
            datetimelong = [int(str(date) + str(time)) for date, time in zip(self.longer.date, self.longer.time)]
            datetime_long = [None] * len(datetime)
            i, j = 0, 0
            while j < len(datetime):
                if i == len(datetimelong) - 1 or datetime[j] < datetimelong[i + 1]:
                    datetime_long[j] = i
                    j += 1
                else:
                    i += 1

            self.datetimelong = datetime_long


        self.capital = 100000
        self.open_position = False
        self.buy_price = 0
        self.buy_qty = 0
        self.sell_price = 0
        self.sell_qty = 0
        self.tradetype = None
        self.trades = pd.DataFrame(columns=['Buy price', 'Sell price', 'Quantity','Trade type', 'PNL', 'Return (%)','Capital','Remark','PNL'])
        self.trades.loc[0] = [0,0,0,'None',0,0,100000,'',0]
        self.remarks = ''
        self.number = 0
        self.win = 0
        self.w = 0
        self.loss = 0
        self.l = 0
    
    def reinitialize(self):
            self.buy_price,self.buy_qty,self.sell_price,self.sell_qty = 0,0,0,0
            self.open_position = False
            if self.risk.type == 'adjusted':
                self.adjusted = 0
            self.tradetype = None

    def long(self,index,qty = 0):
        # Squarring off
        if self.open_position:
            self.buy_price = round(self.stocks.data.price[index],2)
            self.buy_qty = qty
            pnl = round((self.sell_price - self.buy_price) * qty,2)
            pnl_percent = (pnl/self.capital) * 100
            self.capital += pnl
            if pnl>0:
                self.win += 1
                self.w += pnl
            else:
                self.loss +=1
                self.l += pnl
            self.trades.loc[len(self.trades.index)] = [self.buy_price,self.sell_price,qty,self.tradetype,pnl,pnl_percent,self.capital,self.remarks,pnl]
            self.number += 1
            
        else:
            self.buy_price = round(self.stocks.data.price[index],2)
            
            self.buy_qty = self.capital // self.buy_price
            self.open_position = True
            self.tradetype = 'long'
            if self.risk.type == 'atr':
                self.take_profit = 1000000
                self.stop_loss = 0
            elif self.risk.type == 'adjusting':
                self.take_profit = [1000000]*4
                self.stop_loss = [0] * 4
            self.stop_loss = self.risk.set_stop_loss(self.sell_price,index,self.tradetype,self.stop_loss)
            self.take_profit = self.risk.set_take_profit(self.sell_price,index,self.tradetype,self.take_profit)
            if self.risk.type == 'adjusting':
                self.adjusted = self.buy_qty // 4
            self.number += 1
    

    def short(self,index,qty = 0):
        # Squarring off
        if self.open_position:
            self.sell_price = round(self.stocks.data.price[index],2)
            self.sell_qty = qty
            pnl = round((self.sell_price - self.buy_price) * qty,2)
            pnl_percent = (pnl/self.capital) * 100
            self.capital += pnl
            if pnl>0:
                self.win += 1
                self.w += pnl
            else:
                self.loss +=1
                self.l += pnl
            self.trades.loc[len(self.trades.index)] = [self.buy_price,self.sell_price,qty,self.tradetype,pnl,pnl_percent,self.capital,self.remarks,pnl]
            
        else:
            self.sell_price = round(self.stocks.data.price[index],2)
            self.sell_qty = self.capital // self.sell_price
            self.open_position = True
            self.tradetype = 'short'
            if self.risk.type == 'atr':
                self.take_profit = 0
                self.stop_loss = 1000000
            elif self.risk.type == 'adjusting':
                self.take_profit = [0]*4
                self.stop_loss = [1000000] * 4
            self.stop_loss = self.risk.set_stop_loss(self.sell_price,index,self.tradetype,self.stop_loss)
            self.take_profit = self.risk.set_take_profit(self.sell_price,index,self.tradetype,self.take_profit)
            if self.risk.type == 'adjusting':
                self.adjusted = self.sell_qty // 4
            self.number += 1
    
    
    def condition(self,index):
        # detect regular bullish divergence in last 3 candles ie rsi low gradient > 0 stock, pivot low gradient < 0
        # confirm if k is greater than d 
        if self.open_position:
            pass
        elif self.strategy == 'single':
            if self.stocks.data.regular_bullish_divergence[index] == 1 and self.stocks.data.K[index] > self.stocks.data.D[index] and self.stocks.data.K[index] <20:
                self.long(index)
            # detect regular bullish divergence in last 5 candles ie rsi high gradient < 0 stock, pivot high gradient > 0
            # confirm if k is greater than d 
            if self.stocks.data.regular_bearish_divergence[index] == 1 and self.stocks.data.K[index] < self.stocks.data.D[index] and self.stocks.data.K[index] >80:
                self.short(index)
                
        elif self.strategy == 'multi':
            long_index = int(self.datetimelong[index])
            if self.longer.data['hidden_bullish_divergence'].iloc[long_index] == 1:
                #if self.stocks.data.regular_bullish_divergence[index] == 1 and self.stocks.data.Close > self.stocks.data.EMA50[index]:
                    self.long(index)
            if self.longer.data['hidden_bearish_divergence'].iloc[long_index] == 1:
                #if self.stocks.data.regular_bearish_divergence[index] == 1 and self.stocks.data.Close < self.stocks.data.EMA50[index]:
                    self.short(index)
            
                
                
    
    def check(self,index):
        if self.risk.type == 'atr':
            if self.tradetype == 'long':
                #if self.stocks.data.K[index] < self.stocks.data.D[index] and self.stocks.data.K[index] and self.stocks.data.price[index] > self.buy_price:
                #    self.short(index,self.buy_qty)
                #    self.reinitialize()
                #    self.remarks = 'tp'
                if self.stop_loss > self.stocks.data.Close[index]:
                    self.short(index,self.buy_qty)
                    self.reinitialize()
                    self.remarks = 'sl'
                elif self.take_profit < self.stocks.data.Close[index]:
                    self.short(index,self.buy_qty)
                    self.reinitialize()
                    self.remarks = 'tp'
            elif self.tradetype == 'short':
                #if self.stocks.data.K[index] > self.stocks.data.D[index] and self.stocks.data.K[index] <20 and self.stocks.data.price[index] < self.sell_price:
                #    self.long(index,self.sell_qty)
                #    self.reinitialize()
                #    self.remarks = 'tp'
                if self.stop_loss < self.stocks.data.Close[index]:
                    self.long(index,self.sell_qty)
                    self.reinitialize()
                    self.remarks = 'sl'
                elif self.take_profit > self.stocks.data.Close[index]:
                    self.long(index,self.sell_qty)
                    self.reinitialize()
                    self.remarks = 'tp'
        
        elif self.risk.type == 'adjusting':
            
            if self.tradetype == 'long':
                # check stoploss
                for i in range(len(self.stop_loss)-1):
                    if self.stop_loss[i] != 0 and self.stop_loss[i] > self.stocks.data.Close[index]:
                        self.short(index, self.adjusted)
                        self.stop_loss[i] = 0
                        self.remarks = f"sl{i+1}"
                if self.stop_loss[-1] != 0 and self.stop_loss[-1] > self.stocks.data.Close[index]:
                    self.remarks = 'sl4'
                    self.short(index, self.buy_qty - 3*(self.adjusted))
                    self.reinitialize()
                # check take profit
                for i in range(len(self.take_profit)-1):
                    if self.take_profit[i] != 0 and self.take_profit[i] < self.stocks.data.Close[index]:
                        self.short(index, self.adjusted)
                        self.take_profit[i] = 0
                        self.remarks = f"tp{i+1}"
                if self.take_profit[-1] != 0 and self.take_profit[-1] < self.stocks.data.Close[index]:
                    self.remarks = 'tp4'
                    self.short(index, self.buy_qty - 3*(self.adjusted))
                    self.reinitialize()
            if self.tradetype == 'short':
                for i in range(len(self.stop_loss)-1):
                    if self.stop_loss[i] != 0 and self.stop_loss[i] < self.stocks.data.Close[index]:
                        self.long(index, self.adjusted)
                        self.stop_loss[i] = 0
                        self.remarks = f"sl{i+1}"
                if self.stop_loss[-1] != 0 and self.stop_loss[-1] < self.stocks.data.Close[index]:
                    self.remarks = 'sl4'
                    self.long(index, self.sell_qty - 3*(self.adjusted))
                    self.reinitialize()
                for i in range(len(self.take_profit)-1):
                    if self.take_profit[i] != 0 and self.take_profit[i] > self.stocks.data.Close[index]:
                        self.long(index, self.adjusted)
                        self.take_profit[i] = 0
                        self.remarks = f"tp{i+1}"
                if self.take_profit[-1] != 0 and self.take_profit[-1] > self.stocks.data.Close[index]:
                    self.remarks = 'tp4'
                    self.long(index, self.sell_qty - 3*(self.adjusted))
                    self.reinitialize()
                


    def run_strategy(self):
        for i in range(len(self.stocks.data.Close)):
            if self.open_position:
                if self.tradetype == 'long':
                    self.stop_loss = self.risk.update_stop_loss(self.buy_price,i,self.tradetype,self.stop_loss)
                    self.take_profit = self.risk.update_take_profit(self.buy_price,i,self.tradetype,self.take_profit)
                    self.check(i)
                elif self.tradetype == 'short':
                    self.stop_loss = self.risk.update_stop_loss(self.sell_price,i,self.tradetype,self.stop_loss)
                    self.take_profit = self.risk.update_take_profit(self.sell_price,i,self.tradetype,self.take_profit)
                    self.check(i)
            else:
                self.condition(i)
    
        try:
            summary.loc[len(summary)] = [self.stocks.ticker,round(self.capital-100000,2),self.number,round(((self.capital-100000)/100000)*100,2),round(((self.win)/(self.win+self.loss))*100,2),round(((self.w)/(self.win))*100,2),round(((self.l)/(self.loss))*100,2)]
        except Exception as e:
            print(e)
        self.trades.to_csv(f"rsi_divergence/trades/{self.stocks.ticker}_{self.strategy}_{self.timeframe}_{self.risk_strategy}_trades.csv")
        print(self.__repr__())

    def __repr__(self):
        self.end_time = time.time()
        return f"| Ticker ==> {self.stocks.ticker} | p/l ==>  {round(self.capital-100000,2)} ( {round(((self.capital-100000)/100000)*100,2)} % ) | time taken ==>  {round(self.end_time-self.start_time,2)} s |"
ticker = ['BHARTIARTL.NS','BAJFINANCE.NS','HDFCLIFE.NS','TITAN.NS','BAJAJ-AUTO.NS','KOTAKBANK.NS','ONGC.NS','HINDALCONS','ADANIENT.NS','TATASTEELINS','NTPC.NS','CIPLA.NS','LTIM.NS','APOLLOHOSP.NS','BAJAJFINSV.NS','NESTLEIND.NS','ITC.NS','TCS.NS','INDUSINDBK.NS','TATACONSUM.NS','RELIANCE.NS','BRITANNIA.NS','MARUTI.NS','ULTRACEMCO.NS','LT.NS','COALINDIA.NS','WIPRO.NS','HEROMOTOCO.NS','SHRIKRAMFIN.NS']

tsla = strategy('TSLA','atr','multi','5m')
tsla.run_strategy()
tickers = pd.ExcelFile('rsi_divergence/tickers.xlsx').parse('Complete Stock List')['Ticker'][:100]

# download data
progress_bar = tqdm(tickers)
for ticker in tickers:
    try:
        stock_short = Stocks(ticker,'5m')
        stock_long = Stocks(ticker,'30m')
        
    except Exception as e:
        print(e)
    progress_bar.update()
summary.to_csv("rsi_divergence/summary/summary_5m.csv")




# run strategy 
#progress_bar = tqdm(tickers)
#
#for ticker in tickers:
#    try:
#        tick = strategy(ticker,'atr')
#        tick.run_strategy()
#    except Exception as e:
#        print(e)
#    progress_bar.update()
#summary.to_csv("rsi_divergence/summary/summary_5m.csv")
#
#