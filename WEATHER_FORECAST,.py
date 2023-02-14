#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px


# In[3]:


data = pd.read_csv("DailyDelhiClimateTrain.csv")


# In[4]:


data.head()


# In[5]:


data.info()


# In[7]:


data.isnull().sum()


# In[11]:


data.describe()


# In[15]:


figure = plt.scatter(data = data ,x = 'date',y = "meantemp")


# In[16]:


figure = px.line(data, x="date", 
                 y="meantemp", 
                 title='Mean Temperature in Delhi Over the Years')
figure.show()


# In[17]:


forecast_data = data.rename(columns = {"date": "ds", 
                                       "meantemp": "y"})
print(forecast_data)


# In[19]:


from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


# In[21]:


model = Prophet().fit(forecast_data)


# In[25]:


forecasts = model.make_future_dataframe(periods=250)
predictions = model.predict(forecasts)
plot_plotly(model, predictions)


# In[23]:


test_data = pd.read_csv("DailyDelhiClimateTest.csv")


# In[24]:


figure = px.line(test_data, x="date", 
                 y="meantemp", 
                 title='Mean Temperature in Delhi Over the Years')
figure.show()


# In[ ]:




