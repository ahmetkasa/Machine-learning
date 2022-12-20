#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


# In[2]:


data = pd.read_csv("Hitters.csv")
df = data.copy()
df.head()


# In[3]:


df = df.dropna()


# In[4]:


df.isnull().sum()


# In[5]:


dms = pd.get_dummies(df[["League","Division","NewLeague"]])


# In[6]:


dms.head()


# In[7]:


y = df["Salary"]


# In[8]:


X_ = df.drop(["Salary","League","Division","NewLeague"],axis = 1)


# In[9]:


X = pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis = 1)


# In[10]:


X.head()


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

print("X_train shape =",X_train.shape)
print("X_test shape =",X_test.shape)
print("y_train shape =",y_train.shape)
print("y_test shape =",y_test.shape)


# In[13]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# In[14]:


pca = PCA()


# In[15]:


X_reducet_train = pca.fit_transform(scale(X_train))


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


lm = LinearRegression()


# In[18]:


pcr_model = lm.fit(X_reducet_train,y_train)


# In[19]:


pcr_model.intercept_


# In[20]:


pcr_model.coef_


# In[21]:


#model predict 


# In[22]:


y_pred = pcr_model.predict(X_reducet_train)


# In[23]:


y_pred[:5]


# In[27]:


r2_score(y_train,y_pred)


# In[ ]:





# In[28]:


pca2 = PCA()


# In[29]:


X_reduced_test = pca2.fit_transform(scale(X_test))


# In[30]:


y_pred = pcr_model.predict(X_reduced_test)


# In[31]:


from sklearn.metrics import mean_squared_error


# In[32]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[33]:


#MODEL TUNİNG 


# In[34]:


lm = LinearRegression()
pcr_model = lm.fit(X_reducet_train,y_train)
y_pred = pcr_model.predict(X_reduced_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))


# In[35]:


from sklearn import model_selection


# In[37]:


cv_10 = model_selection.KFold(n_splits=10,shuffle=True,random_state=1)


# In[38]:


lm = LinearRegression()


# In[39]:


RMSE = []


# In[49]:


for i in np.arange(1,X_reducet_train.shape[1]+1):
    score = np.sqrt(-1*model_selection.cross_val_score(lm,
                                                      X_reducet_train[:,:i],
                                                       y_train.ravel()
                                                      ,cv = cv_10,
                                                      scoring = "neg_mean_squared_error"))
    
    RMSE.append(score)


# In[50]:


import matplotlib.pyplot as plt


# In[51]:


plt.plot(RMSE,'-v')
plt.xlabel("bileşen sayısı")
plt.ylabel("RMSE")
plt.title("maaş tahmin modeli")


# In[ ]:




