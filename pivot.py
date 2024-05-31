
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


class Pivot:
    def __init__(self,data):
        self.data = data
    # time period =3, multiplier 2
    def atr(self,timeperiod=21):
        self.data['atr'] = ta.atr(self.data['High'], self.data['Low'], self.data['Close'], 21) * 2
        

    def directional_change(self,close, high, low, sigma):
        up_zig = True  # Last extreme is a bottom. Next is a top. 
        tmp_max = high[0]
        tmp_min = low[0]
        tmp_max_i = 0
        tmp_min_i = 0

        tops = []
        bottoms = []

        for i in range(len(close)):
            if up_zig:  # Last extreme is a bottom
                if high[i] > tmp_max:
                    # New high, update
                    tmp_max = high[i]
                    tmp_max_i = i
                elif close[i] < tmp_max - sigma[i]: 
                    # Price retraced by sigma %. Top confirmed, record it
                    top = [i, tmp_max_i, tmp_max]
                    tops.append(top)

                    # Setup for next bottom
                    up_zig = False
                    tmp_min = low[i]
                    tmp_min_i = i
            else:  # Last extreme is a top
                if low[i] < tmp_min:
                    # New low, update
                    tmp_min = low[i]
                    tmp_min_i = i
                elif close[i] > tmp_min + sigma[i]: 
                    # Price retraced by sigma %. Bottom confirmed, record it
                    bottom = [i, tmp_min_i, tmp_min]
                    bottoms.append(bottom)

                    # Setup for next top
                    up_zig = True
                    tmp_max = high[i]
                    tmp_max_i = i

        return tops, bottoms

    def get_extremes(self):
        self.atr(self.data)
        tops, bottoms = self.directional_change(self.data['Close'], self.data['High'], self.data['Low'], self.data['atr'])
        tops = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
        bottoms = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
        tops['type'] = 1
        bottoms['type'] = -1
        extremes = pd.concat([tops, bottoms])
        extremes = extremes.set_index('conf_i')
        extremes = extremes.sort_index()
        return extremes

    def plot(self):
        self.data['date'] = self.data.index.astype('datetime64[s]')
        self.data = self.data.set_index('date')
        self.atr(self.data)
        
        tops, bottoms = self.directional_change(self.data['Close'].to_numpy(), self.data['High'].to_numpy(), self.data['Low'].to_numpy(), self.data['atr'].to_numpy())
    
        # Plot the closing price
        plt.figure(figsize=(14, 7))
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue')
    
        # Combine and sort tops and bottoms
        extremes = sorted(tops + bottoms, key=lambda x: x[0])
    
        # Plot lines connecting each top to the next bottom and each bottom to the next top
        for i in range(len(extremes) - 1):
            plt.plot([self.data.index[extremes[i][1]], self.data.index[extremes[i + 1][1]]], [extremes[i][2], extremes[i + 1][2]], color='gray', linestyle='--')
    
        # Plot tops and bottoms
        for top in tops:
            plt.plot(self.data.index[top[1]], top[2], marker='o', color='green', markersize=4, label='Top' if top == tops[0] else "")
        for bottom in bottoms:
            plt.plot(self.data.index[bottom[1]], bottom[2], marker='o', color='red', markersize=4, label='Bottom' if bottom == bottoms[0] else "")
    
        plt.legend()
        plt.title('Directional Changes with Tops and Bottoms')      
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

    

#pivots = Pivot(GOOG)
#pivots.plot()
#print(pivots.get_extremes())
