#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt


# In[3]:


ad = pd.read_csv("Advertising.csv")
df = ad.copy()


# In[4]:


df = df.iloc[:,1:len(df)]


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()#eksik değer yok 


# In[8]:


df.describe().T


# In[9]:


df.corr() #değişkenler arasındaki korelasyonlar 


# In[10]:


sns.pairplot(df,kind="reg")


# In[11]:


sns.jointplot(x= "TV",y="sales",data = df,kind = "reg")


# In[12]:


import statsmodels.api as sm


# In[13]:


X= df[["TV"]]
X[0:5]


# In[14]:


X= sm.add_constant(X)


# In[15]:


X[0:5]


# In[16]:


y = df[["sales"]]


# In[17]:


y[0:5]


# In[18]:


lm = sm.OLS(y,X)


# In[19]:


model = lm.fit()


# In[20]:


model.summary()


# In[21]:


model.params


# In[22]:


model.summary().tables[1]


# In[23]:


model.mse_model


# In[24]:


model.rsquared


# In[25]:


model.rsquared_adj


# In[26]:


model.fittedvalues[0:5]


# In[27]:


y[0:5]


# In[32]:


plt.plot(y,X)


# In[33]:


g = sns.regplot(df["TV"],df["sales"])


# In[ ]:




