import pandas as pd 
import math
import quandl
import numpy as numpy
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = 'TKncTNa-1viiEudz7Uxx'
df = quandl.get_table('WIKI/PRICES')
# print(df)
df = df [['adj_open','adj_high','adj_low','adj_close','adj_volume',]]
df['HL_PCT'] = (df['adj_high']-df['adj_close'])/df['adj_close']*100.0
df['PCT_change'] = (df['adj_close']-df['adj_open'])/df['adj_open']*100.0

df = df[['adj_close','HL_PCT','PCT_change','adj_volume']]
forecast_col = 'adj_close'
df.fillna(-9999999,inplace=True)

forecast_out = int (math.ceil(0.1*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())
# print(df.head())