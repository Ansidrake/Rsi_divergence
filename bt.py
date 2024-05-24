
from backtesting import Strategy
from backtesting import Backtest
import pandas as pd
import main

class rsi_divergence(Strategy):
    stock = main.Stocks('HDFCLIFE.NS','1h')
    rsi_period = 14
    atr_period = 10
    atr_multiplier = 2
    lbp = 5

    def regular_bullish_divergence(self):
        self.stock.regular_bullish_divergence()
        return self.stock.data.regular_bullish_divergence
    def regular_bearish_divergence(self):
        self.stock.regular_bearish_divergence()
        return self.stock.data.regular_bearish_divergence
    def rsi(self):
        self.stock.rsi(self.rsi_period)
    def stochD(self):
        self.stock.stoch()
        return self.stock.data.D
    def stochK(self):
        self.stock.stoch()
        return self.stock.data.K
    def atr(self):
        self.stock.atr(self.atr_period)
        return self.atr_multiplier * self.stock.data.atr
        
    def init(self):
        super().init
        df = pd.read_csv('rsi_divergence/cache/HDFCLIFE.NS_1h_data.csv', parse_dates=['Datetime'], index_col='Datetime')

        self.regular_bullish_divergence = self.I(self.regular_bullish_divergence)
        self.regular_bearish_divergence = self.I(self.regular_bearish_divergence)
        self.D = self.I(self.stochD)
        self.K = self.I(self.stochK)
        self.atr = self.I(self.atr)
        
            
    def next(self):
        if self.position:
            for trade in self.trades:
                if trade.is_long:
                    trade.sl = self.data.Close[-1] - self.atr[-1] * 2
                    trade.tp = self.data.Close[-1] + self.atr[-1] * 2
                elif trade.is_short:
                    trade.sl = self.data.Close[-1] + self.atr[-1] * 2
                    trade.tp = self.data.Close[-1] - self.atr[-1] * 2
    
        if self.regular_bullish_divergence[-1] == 1 and self.K[-1] > self.D[-1] and self.K[-1] < 20:
            self.buy(size=1,sl = self.data.Close[-1] - self.atr[-1]*2,tp = self.data.Close[-1] + self.atr[-1]*2)
        
        if self.regular_bearish_divergence[-1] == 1 and self.K[-1] < self.D[-1] and self.K[-1] > 80:
            self.sell(size=1,sl = self.data.Close[-1] + self.atr[-1]*2,tp = self.data.Close[-1] - self.atr[-1]*2)

data = pd.read_csv('rsi_divergence/cache/HDFCLIFE.NS_1h_data.csv', parse_dates=['Datetime'], index_col='Datetime')

bt = Backtest(pd.read_csv('rsi_divergence/cache/HDFCLIFE.NS_1h_data.csv'), rsi_divergence, cash=10000)
stats = bt.run()
#stats = bt.optimize(
#    rsi_period = range(14,43,7),
#    atr_period = range(2,21,3),
#    atr_multiplier = range(1,6),
#    lbp = range(5,16,5),
#    maximize= 'Return [%]'
#    )
print(stats)
bt.plot()


