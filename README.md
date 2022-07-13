# Machine-learning LSTM MODEL 
import pandas as pd 
import numpy as np
import math 
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

#read data 
df = pd.read_csv("/EREGL.IS.csv")
df.head()

plt.figure(figsize=(18,5))
plt.title('EREGL')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=15 )
plt.ylabel('FİYAT)', fontsize=15)
plt.show()



data = df.filter(['Close'])

dataset = data.values

from keras.engine import training
training_data_len = math.ceil(len(dataset)*0.85)
training_data_len
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


train_data = scaled_data[0:training_data_len  , : ]

x_train=[]
y_train = []
for i in range(180,len(train_data)):
    x_train.append(train_data[i-180:i,0])
    y_train.append(train_data[i,0])

    if i<=180:
      print(x_train)
      print(y_train)
      print()
      
      
      
 x_train,y_train = np.array(x_train),np.array(y_train)
 
 x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
 
 model = Sequential()
model.add(LSTM(units = 50,return_sequences=True , input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam',loss = 'mean_squared_error')

model.fit(x_train,y_train,batch_size = 1,epochs = 1)

test_data = scaled_data[training_data_len - 180: , : ]

x_test = []
y_test =  dataset[training_data_len : , : ] 
for i in range(180,len(test_data)):
    x_test.append(test_data[i-180:i,0])
    
    


test_data = scaled_data[training_data_len - 180: , : ]

x_test = []
y_test =  dataset[training_data_len : , : ] 
for i in range(180,len(test_data)):
    x_test.append(test_data[i-180:i,0])
    
    

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)
#Undo scaling


#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse



train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


plt.figure(figsize=(20,5))
plt.title('EREGL')
plt.xlabel('zaman', fontsize=8)
plt.ylabel('FİYAT', fontsize=12)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend([ 'GERÇEK DEĞER ', 'TAHMİN'], loc='lower right')
plt.show()
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)
#Undo scaling


#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse



train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


plt.figure(figsize=(20,5))
plt.title('EREGL')
plt.xlabel('zaman', fontsize=8)
plt.ylabel('FİYAT', fontsize=12)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend([ 'GERÇEK DEĞER ', 'TAHMİN'], loc='lower right')
plt.show()
