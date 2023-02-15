#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px 
from sklearn.model_selection import train_test_split

from sklearn.linear_model import PassiveAggressiveRegressor


# In[27]:


data = pd.read_csv("ank.csv")


# In[28]:


data.head(10)


# In[29]:


data.isnull().sum()


# In[30]:


print(data.type.value_counts())


# In[31]:


correlation = data.corr()


# In[32]:


print(correlation["isFraud"].sort_values(ascending=False))


# In[33]:


data["type"] = data["type"].map({"CASH_OUT":1,"PAYMENT":2,"CASH_IN":3,"TRANSFER":4,"DEBIT":5})


# In[34]:


data.head()


# In[35]:


data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})


# In[36]:


data.head()


# In[41]:


X = np.array(data[["type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]])
y = np.array(data["isFraud"])


# In[42]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10,random_state=42)


# In[43]:


from sklearn.tree import DecisionTreeClassifier


# In[44]:


model = DecisionTreeClassifier().fit(X_train,y_train)


# In[45]:


model.score(X_test,y_test)


# In[49]:


#prediction 
features = np.array([[4,1864.28,41554.0,0.00,21182.0,0]])
model.predict(features)


# In[ ]:




