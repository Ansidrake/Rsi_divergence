
import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm
import talib 
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

def ema(close):
    return ta.ema(close, length=50)

def rsi(close):
    rsi_values = talib.RSI(close, timeperiod=14)
    return rsi_values.dropna()

def stoch(rsi):
    period = 14
    smoothK = 3
    smoothD = 3
    stochrsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    K = 100 * stochrsi.rolling(smoothK).mean()
    D = K.rolling(smoothD).mean()
    return K, D

def atr(high, low, close):
    return talib.ATR(high, low, close, timeperiod=10)

def pivot(source, type_):
    pivot = [None] * len(source)
    if type_ == 'high':
        for i in range(5, len(source) - 5):
            if source[i] == max(source[i - 5:i + 6]):
                pivot[i] = source[i]
        for i in range(5):
            if source[i] == max(source[:i + 6]):
                pivot[i] = source[i]
            if source[-i-1] == max(source[-(i + 6):]):
                pivot[-i-1] = source[-(i + 6)]
    elif type_ == 'low':
        for i in range(5, len(source) - 5):
            if source[i] == min(source[i - 5:i + 6]):
                pivot[i] = source[i]
        for i in range(5):
            if source[i] == min(source[:i + 6]):
                pivot[i] = source[i]
            if source[-i-1] == min(source[-(i + 6):]):
                pivot[-i-1] = source[-(i + 6)]
    return pivot

def calculate_pivot_indexes(pivot):
    pivot_index_prev = [0] * len(pivot)
    pivot_index_next = [0] * len(pivot)
    for i in range(len(pivot)):
        if pivot[i] is None and i > 0:
            pivot_index_prev[i] = pivot_index_prev[i - 1]
        else:
            pivot_index_prev[i], pivot_index_next[i] = i, i
    for i in range(len(pivot) - 1, -1, -1):
        flag = False
        if pivot[i] is not None:
            if flag and i < len(pivot) - 1:
                pivot_index_prev[i], pivot_index_next[i] = i, pivot_index_next[i + 1]
            else:
                pivot_index_prev[i], pivot_index_next[i] = i, i
                flag = True
        elif i < len(pivot) - 1:
            pivot_index_next[i] = pivot_index_next[i + 1]
    return pivot_index_prev, pivot_index_next

def gradient(data, name, pivot_index_name_prev, pivot_index_name_next):
    gradient_values = [0] * len(data[name])
    for i in range(len(data[name])):
        last, n = pivot_index_name_prev[i], pivot_index_name_next[i]
        gradient_values[i] = (data[name].iloc[n] - data[name].iloc[last]) / data[name].iloc[last]
    return gradient_values

def regular_bullish_divergence(close, gradient_low_rsi, gradient_low):
    signal = [0] * len(close)
    for index in range(len(close)):
        for i in range(5):
            if gradient_low_rsi[index - i] > 0 and gradient_low[index - i] < 0:
                signal[index] = 1
    return signal

def hidden_bullish_divergence(close, gradient_low_rsi, gradient_low):
    signal = [0] * len(close)
    for index in range(len(close)):
        if gradient_low_rsi[index] < 0 and gradient_low[index] > 0:
            signal[index] = 1
    return signal

def regular_bearish_divergence(close, gradient_high_rsi, gradient_high):
    signal = [0] * len(close)
    for index in range(len(close)):
        for i in range(5):
            if gradient_high_rsi[index - i] < 0 and gradient_high[index - i] > 0:
                signal[index] = 1
    return signal

def hidden_bearish_divergence(close, gradient_high_rsi, gradient_high):
    signal = [0] * len(close)
    for index in range(len(close)):
        if gradient_high_rsi[index] > 0 and gradient_high[index] < 0:
            signal[index] = 1
    return signal


