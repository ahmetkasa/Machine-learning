# -*- coding: utf-8 -*-
"""Predicting Stock Prices Using Monte Carlo Methods in Python.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t8DjofHpGVit-wFIeOyf75aqqwBjk7Ff
"""

import pandas as pd 
import numpy as np 
from random import random   
import matplotlib.pyplot as plt 
from scipy.stats import norm

hist = pd.read_csv("/XU100.IS.csv")

hist.head()

hist = hist[['Close']]

hist.head()

hist['Close'].plot(title="SİSE HİSSE FİYATI", ylabel=
                   "KAPANIŞ FİYATI ", figsize=[20, 12])


plt.grid()

# Create Day Count, Price, and Change Lists
days = [i for i in range(1, len(hist['Close'])+1)]
price_orig = hist['Close'].tolist()
change = hist['Close'].pct_change().tolist()
change = change[1:]  # Removing the first term since it is NaN

# Statistics for Use in Model
mean = np.mean(change)
std_dev = np.std(change)
print('\nOrtalama yüzde değişim: ' + str(round(mean*100, 2)) + '%')
print('Yüzde değişimin Standart Sapması: ' +   
      str(round(std_dev*100, 2)) + '%')

start_date = '2012-05-04'
end_date = '2022-05-04'

# Simülasyon Sayısı ve Tahmin Dönemi
simulations = 200 # yani 200 adet tahmin yapıp karar verecek ,sayı ne kadar çok olursa o kadar iyi 
days_to_sim = 1*252 # 1 yıldaki işlem sayısı

# Simülasyon için Şekil Başlatma
fig = plt.figure(figsize=[10, 6])
plt.plot(days, price_orig)
plt.title("Monte Carlo Stock Prices [" + str(simulations) + 
          " simulations]")
plt.xlabel("Trading Days After " + start_date)
plt.ylabel("Closing Price [$]")
plt.xlim([2000, len(days)+days_to_sim])
plt.grid()

#Analiz için Listeleri Başlatma
close_end = []
above_close = []

# İstenen Simülasyon Sayısı için Döngü İçin
for i in range(simulations):
    num_days = [days[-1]]
    close_price = [hist.iloc[-1, 0]]
    
    # For Loop for Number of Days to Predict
    for j in range(days_to_sim):
        num_days.append(num_days[-1]+1)
        perc_change = norm.ppf(random(), loc=mean, scale=std_dev)
        close_price.append(close_price[-1]*(1+perc_change))

    if close_price[-1] > price_orig[-1]:
        above_close.append(1)
    else:
        above_close.append(0)

    close_end.append(close_price[-1])
    plt.plot(num_days, close_price)

