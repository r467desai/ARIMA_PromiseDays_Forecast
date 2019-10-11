#!/usr/bin/env python
# coding: utf-8

# In[45]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import random


# In[46]:


df = pd.read_excel("Item103478.xlsx")
df['CREATION DATE'].min(), df['CREATION DATE'].max()


# In[47]:


cols = ['CUSTOMER NAME',          
'SHIP TO',         
'DEST ORG',          
'SALES ORDER TYPE',
'VENDOR_NAME',       
'PO #',              
'SO #',              
'CUSTOMER PO',       
'REQ #',             
'PO STATUS',         
'PO LINE',           
'Product GBU',       
'ORDERED ITEM',      
'ITEM #',            
'SUPPLIER_ITEM',     
'ITEM DESCRIPTION',    
'RETD',              
'NEED BY DATE',
'POD',               
'POA',               
'TRANSIT DAYS',      
'APPR STATUS',       
'CURR',                        
'PO UOM',            
'QTY ORDERED',       
'QTY CANCELLED',     
'QTY PO BALANCE',    
'PROMISED DATE',     
'CETD',              
'ATD',               
'BOL #',             
'CONTAINER #',       
'LOT_NUM',           
'SHIP LINE STATUS']
df.drop(cols, axis=1, inplace=True)
df = df.sort_values('CREATION DATE')
df.isnull().sum()


# In[48]:


cols=['PRICE','ER / ISP SHIPMENT #','ER/SHIPPED QTY']
df.drop(cols, axis=1, inplace=True)
df = df.sort_values('CREATION DATE')
df.isnull().sum()


# In[49]:


df.fillna(df.mean(), inplace=True)


# In[50]:


df.dtypes


# In[51]:


df['CREATION DATE'] = pd.to_datetime(df['CREATION DATE'])


# In[52]:


df.dtypes


# In[53]:


df = df.set_index('CREATION DATE')
df.index


# In[65]:


y = df['PromisedDays'].resample('M').mean()


# In[66]:


y['2017':]


# In[67]:


y.plot(figsize=(15, 6))
plt.show()


# In[57]:


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()


# In[58]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[59]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[60]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[68]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[69]:


pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2015':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='B', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Promise Days')
plt.legend()
plt.show()


# In[70]:


y_forecasted = pred.predicted_mean
y_truth = y['2019-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[71]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# In[72]:


pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Promise Days')
plt.legend()
plt.show()

