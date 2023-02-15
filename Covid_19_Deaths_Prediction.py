#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# In[3]:


data = pd.read_csv("covıd.csv")


# In[7]:


data.head()


# In[12]:


plt.scatter(data = data,x = "Date",y = "Daily Confirmed")


# In[15]:


data.isnull().sum()


# In[18]:


data = data.drop("Date",axis = 1)


# In[19]:


data.head()


# In[20]:


import plotly.express as px
fig = px.bar(data, x='Date_YMD', y='Daily Confirmed')
fig.show()


# In[22]:


pip install autots


# In[23]:


from autots import AutoTS
model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple')
model = model.fit(data, date_col="Date_YMD", value_col='Daily Deceased', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)


# In[ ]:




