import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm
import talib 

class Stocks:
    
    def __init__(self,ticker):
        self.ticker = ticker
        self.historical_data()
        #try:
        #    self.data = self.load_data()
        #except:
        #    print('not using cache')
        #    self.historical_data()
        #    self.data.to_csv(f"rsi_divergence/cache/{self.ticker}_data.csv")  
        
        print(self.data)

    def load_data(self):
        filename = f"rsi_divergence/cache/{self.ticker}_data.csv"  

        return pd.read_csv(filename, index_col=0)

    def historical_data(self):

        df_list = []
        data = yf.download(self.ticker, group_by="Ticker", period='max',progress=False)
        df_list.append(data)
        df = pd.concat(df_list)
        self.data = df
        self.data['price'] = self.data.Open.shift(-1)
        self.data = self.data.dropna()
        self.rsi()
        self.pivot_high()
        self.stoch()
        self.data = self.data.dropna()
        print(self.data)

    def rsi(self):
        self.data['rsi'] = talib.RSI(self.data.Close, timeperiod=14)
        self.data = self.data.dropna()
        #print(self.data)
    
    def stoch(self):
        self.data['k'],self.data['d'] = talib.STOCH(self.data.rsi,self.data.rsi, self.data.rsi, 14)

    def pivot_high(self):
        pass

    def pivot_low(self):
        pass

    def pivot_rsi_high(self):
        pass

    def pivot_rsi_low(self):
        pass
    

    

Tsla = Stocks('TSLA')
    

summary = pd.DataFrame(columns=['Ticker', 'P/L', 'No.of trades', 'Return (%)'])
        
class strategy:
    def __init__(self,ticker):
        self.start_time = time.time()
        stocks =  Stocks(ticker)
        self.capital = 10000
        self.open_position = False
        self.buy_price = 0
        self.buy_qty = 0
        self.sell_price = 0
        self.sell_qty = 0
        self.tradetype = None
        self.trades = pd.DataFrame(columns=['Buy price', 'Sell price', 'Quantity','Trade type', 'PNL', 'Return (%)','Capital'])
        self.trades.loc[0] = [0,0,0,'None',0,0,10000]
        self.stocks = stocks
        self.number = 0
        #print(self.stocks.data.iloc[0]['Close'])
        #print('success')

    
    def long(self,index):
        # Squarring off
        if self.open_position:
            self.buy_price = self.stocks.data.iloc[index]['Close']
            self.buy_qty = self.sell_qty
            pnl = (self.sell_price - self.buy_price) * self.sell_qty
            pnl_percent = (pnl/self.capital) * 100
            self.capital += pnl
            self.trades.loc[len(self.trades.index)] = [self.buy_price,self.sell_price,self.sell_qty,self.tradetype,pnl,pnl_percent,self.capital]
            # reinitializing the conditions
            self.buy_price,self.buy_qty,self.sell_price,self.sell_qty = 0,0,0,0
            self.open_position = False
            self.tradetype = None
            self.number += 1
        else:
            self.buy_price = self.stocks.data.iloc[index]['Close']
            if self.capital < self.buy_price:
                pass
            else:
                self.buy_qty = self.capital // self.buy_price
            self.open_position = True
            self.tradetype = 'long'
            self.number += 1

    
    def short(self,index):
        # Squarring off
        if self.open_position:
            self.sell_price = self.stocks.data.iloc[index]['Close']
            self.sell_qty = self.buy_qty
            pnl = (self.sell_price - self.buy_price) * self.sell_qty
            pnl_percent = (pnl/self.capital) * 100
            self.capital += pnl
            self.trades.loc[len(self.trades.index)] = [self.buy_price,self.sell_price,self.sell_qty,self.tradetype,pnl,pnl_percent,self.capital]
            # reinitializing the conditions
            self.buy_price,self.buy_qty,self.sell_price,self.sell_qty = 0,0,0,0
            self.open_position = False
            self.tradetype = None
            self.number += 1
        else:
            self.sell_price = self.stocks.data.iloc[index]['Close']
            if self.capital < self.buy_price:
                pass
            else:
                self.sell_qty = self.capital // self.sell_price
            self.open_position = True
            self.tradetype = 'short'
            self.number += 1

    def condition(self,index):
        if self.stocks.data.iloc[index]['ema20'] > self.stocks.data.iloc[index]['ema50'] and self.stocks.data.iloc[index-1]['ema20'] <= self.stocks.data.iloc[index-1]['ema50']:
            if self.open_position:
                #square off
                if self.tradetype == 'short':
                    self.long(index)
            else:
                # take new positions
                self.long(index)
        elif self.stocks.data.iloc[index]['ema20'] < self.stocks.data.iloc[index]['ema50'] and self.stocks.data.iloc[index-1]['ema20'] >= self.stocks.data.iloc[index-1]['ema50']:
            if self.open_position:
                if self.tradetype == 'long':
                    #square off
                    self.short(index)
            else:
                # take new positions
                self.short(index)
    
    def run_strategy(self):
        #print('Running')
        for i in range(49,len(self.stocks.data['Close'])):
            self.condition(i)
        #print('Completed')
        print(self.trades)
        #print(self.stocks.data)
        summary.loc[len(summary)] = [self.stocks.ticker,round(self.capital-10000,2),self.number,round(((self.capital-10000)/10000)*100,2)]
        
        print(self.__repr__())

    def __repr__(self):
        self.end_time = time.time()
        return f"| Ticker ==> {self.stocks.ticker} | p/l ==>  {round(self.capital-10000,2)} ( {round(((self.capital-10000)/10000)*100,2)} % ) | time taken ==>  {round(self.end_time-self.start_time,2)} s |"

#tickers = pd.ExcelFile('tickers.xlsx').parse('Complete Stock List')['Ticker'][:100]
 
#print(tickers)
#progress_bar = tqdm(tickers)

#for ticker in tickers:
#    try:
#        tick = strategy(ticker)
#        tick.run_strategy()
#    except Exception as e:
#        print(e)
#    #progress_bar.update()
#summary.to_csv('summary.csv')


#tsla = Stocks('TSLA')
#goog = strategy("GOOG")
##goog.historical_data()
##goog.add_ema()
#goog.run_strategy()

