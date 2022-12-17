#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[23]:


df = pd.read_csv("CarPrice_Assignment.csv")
df = df.copy()


# In[24]:


df.head()


# In[25]:


df.isnull().sum()


# In[26]:


df.corr()


# In[27]:


plt.figure(figsize = (16,10))
sns.heatmap(df.corr(),annot= True)


# In[28]:


df.duplicated().sum() #yinelenen satırları kontrol etmede kullanılır 


# In[29]:


df.head()


# In[30]:


df.CarName=df.CarName.apply(lambda x:x.split(" ")[0])


# In[31]:


df.head()


# In[33]:


df.CarName.unique()


# In[34]:


def correct_name(value):
    dict={"toyouta":"toyota","Nissan":"nissan","maxda":"mazda","vokswagen":"volkswagen","porcshce":"porsche"}
    if value in dict.keys():
        return dict[value]
    else:
        return value


# In[35]:


df.CarName.value_counts(normalize=True)*100


# In[36]:


avg_price_bycarName=df.groupby(["CarName"]).mean()["price"].sort_values()


# In[37]:


df_CarName=pd.DataFrame(avg_price_bycarName)
df_CarName["CarName"]=df_CarName.index
df_CarName.reset_index(drop=True,inplace=True)
df_CarName=df_CarName.iloc[:,[1,0]]
df_CarName


# In[38]:


df.symboling.unique()


# In[39]:


df.symboling.value_counts()


# In[40]:


avg_price_bysymboling=df.groupby(["symboling"]).mean()["price"].sort_values()


# In[41]:


df_sym=pd.DataFrame(avg_price_bysymboling)
df_sym["symboling"]=df_sym.index
df_sym.reset_index(drop=True,inplace=True)
df_sym=df_sym.iloc[:,[1,0]]
df_sym


# In[43]:


df.fueltype.unique()


# In[44]:


df.fueltype.value_counts()


# In[45]:


avg_price_byfueltype=df.groupby(["fueltype"])["price"].mean()


# In[46]:


avg_price_byfueltype


# In[47]:


df.aspiration.unique()
df.aspiration.value_counts()


# In[48]:


avg_price_byasp=df.groupby(["aspiration"])["price"].mean()


# In[49]:


avg_price_byasp


# In[50]:


df.head(1)


# In[52]:


df.doornumber.unique()


# In[53]:


df.doornumber.value_counts()


# In[54]:


avg_rpice_bydrnum=df.groupby(["doornumber"])["price"].mean()


# In[55]:


avg_rpice_bydrnum


# In[56]:


categorical_data=df[["symboling","CarName","fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","cylindernumber"]]


# In[57]:


continous_data=df[["wheelbase",'carlength', 'carwidth', 'carheight', 'curbweight','enginesize','boreratio', 'stroke',
       'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']]



# In[58]:


df=df.drop(["car_ID","CarName","symboling","fuelsystem","stroke","compressionratio","peakrpm"],axis=1)


# In[59]:


df


# In[60]:


df=pd.get_dummies(data=df,columns=["fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber"])


# In[88]:


df.head()


# In[62]:


X = df.drop(["price"],axis = 1)


# In[63]:


y = df["price"]


# In[64]:


scaler = StandardScaler()


# In[65]:


X = scaler.fit_transform(X)


# In[66]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)


# In[68]:


print(X_train.shape)
print(X_test.shape)


# In[69]:


lr = LinearRegression()


# In[70]:


lr.fit(X_train,y_train)


# In[71]:


#predict


# In[72]:


y_predictions = lr.predict(X_test)


# In[73]:


y_predictions.shape


# In[74]:


#performans check


# In[79]:


accuracy_traindata=lr.score(X_train,y_train)
accuracy_traindata #eğitim setinde doğruluk oranı %93


# In[81]:


accuracy_testdata=lr.score(X_test,y_test)
accuracy_testdata #test setinde başarı oranı %85


# In[84]:


MSE=mean_squared_error(y_test,y_predictions)
MSE


# In[85]:


MAE=mean_absolute_error(y_test,y_predictions)
MAE


# In[86]:


R_squared=metrics.r2_score(y_test,y_predictions)


# In[ ]:





# In[ ]:




