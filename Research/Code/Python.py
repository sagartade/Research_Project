#!/usr/bin/env python
# coding: utf-8

# In[1]:

#pip install https://github.com/matplotlib/mpl_finance/archive/master.zip

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



# In[2]:



df = pd.read_excel(r'D:\Sagar\Study\Sem3\Research_Thesis\Research_Project\Research\Code\Dataset\Updated_Ethereum.xlsx')
df = df.reset_index()


# In[3]:


df.head(3)


# In[4]:


from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
df["Date"] = df["Date"].apply(mdates.date2num)
ohlc= df[['Date', 'Open', 'High', 'Low','Close']].copy()
f1, ax = plt.subplots(figsize = (16,6))
# plot the candlesticks
candlestick_ohlc(ax, ohlc.values, width=.6, colorup='green', colordown='red')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# Saving image
plt.show()


# In[5]:

df = df.drop(["Date"], axis=1)


# In[6]:


df.shape


# In[7]:

df.head()

# In[16]:


#df['log_price'] = np.log(df['Close']) #Firstly we should take the logarithmic return from prices


# In[17]:


#df['pct_change'] = df['log_price'].diff() #after that let's take difference


# In[18]:


#df['stdev'] = df['pct_change'].rolling(window=30, center=False).std()
#df['Volatility'] = df['stdev'] * (365**0.5) # Annualize.


# In[8]:


plt.figure(figsize=(16,6))
df['Volatility'].plot()
plt.title("Rolling Volatility With 30 Time Periods By Annualized Standard Deviation")
plt.show()


# In[9]:


df = df.dropna()


# In[10]:


vol = df["Volatility"] * 100

# $$\sigma^2(t) = \alpha \times \sigma^2(t-1) + \beta \times e^2(t-1) + w$$

# In[11]:


from arch import arch_model
#am = arch_model(vol, vol='Garch', p=1, o=0, q=1, dist='Normal')

am = arch_model(vol, vol='Garch', p=1, o=1, q=1, dist='Normal')

#from arch.univariate import EGARCH

#am2 = arch_model(vol, vol='EGARCH', p=1, q=1, dist='Normal')


# In[12]:


res1 = am.fit()
res1.summary()

#res1 = am1.fit()
#res1.summary()



#am1.fit.__code__	.co_varnames

# In[13]:


#df['forecast_vol'] = 0.1 * np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]'])

df1 = pd.DataFrame(columns=['test', 'I'])

df1['test'] = res1.resid 
df1.loc[df1['test'] < 0, 'I'] = 1
df1["I"] = df1["I"].fillna(0)

df['forecast_vol'] = 0.1 * np.sqrt(res1.params['omega'] + res1.params['alpha[1]'] * res1.resid**2 + res1.params['gamma[1]'] * res1.resid**2 * df1['I'] + res1.conditional_volatility**2 * res1.params['beta[1]'] )

# After fitting the GARCH(1,1) model, by the formula above, it is possible to forecast rolling volatility. The last 10 rows of the final form of the data is displayed below.

# In[14]:


df.tail(10)


# As it is expected it is seen in the graph below that, GARCH (1,1) model is a weak learner for such a time series. 

# In[15]:


plt.figure(figsize=(16,6))
df["Volatility"].plot()
df["forecast_vol"].plot()
#df["forecast_vol_gjr"].plot()
plt.title("Real Rolling Volatility vs Forecast by GJR-GARCH(1,1)")
plt.legend()
plt.show()


# In order to measure the performance of the model, __Root Mean Squared Error__ is used and the output of this measure for the last 1000 observations is shown below.
# $$\sum{\sqrt{(\hat{X_i}-X_i)^2}}$$

# In[16]:


def rmse_tr(predictions, targets): return np.sqrt(((predictions - targets) ** 2).mean())
skor = rmse_tr(df.loc[df.index[1000:], 'forecast_vol'], df.loc[df.index[1000:], 'Volatility'])
#skor11 = rmse_tr(df.loc[df.index[1000:], 'forecast_vol_gjr'], df.loc[df.index[1000:], 'Volatility'])
print("Root Mean Squared Error of the GARCH(1,1) model is calculated as ",skor)
#print("Root Mean Squared Error of the GJR-GARCH(1,1) model is calculated as ",skor11)
 

# __Now the question is, by using the outputs of the GARCH (1,1) model as inputs, can we build a strong learner model?__
# To find the answer of this question, we will use __Recurrent Neural Networks__ and the GARCH (1,1) outputs will be inputs along with real rolling volatilities

# ## LSTM
# __Firstly it is necessary to forecast volatility by the real rolling volatilities to be able to measure the performance two different models alone.__

# In[22]:


df.shape


# In[23]:


training_set = df.iloc[:, 10:11].values
# 100 timestep ve 1 çıktı ile data yapısı oluşturalım
X_train = []
y_train = []
for i in range(1000, df.shape[0]):X_train.append(training_set[i-1000:i,0])
for i in range(1000, df.shape[0]):y_train.append(training_set[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[24]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[25]:


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[65]:


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.1))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10))
regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# In[66]:


regressor.save('my_modelp1.h5')
 

# In[27]:


from keras.models import load_model
regressor = load_model('my_modelp1.h5')


# In[28]:


predicted_stock_price = regressor.predict(X_train)


# In[44]:


# Visualising the results
plt.figure(figsize=(18,6))
plt.plot(df.iloc[1000:, 10:11].values, color = 'red', label = 'Observed Volatility')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Volatility By LSTM')
plt.title('Real Rolling Volatility vs Forecast of LSTM')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()


# Even though we did not try to build a state of the art LSTM model, yet it is quite impressive, especially on the graphic. However, the graphic might be illusional because we had too many data points in it. That's why we should again measure the goodnes of the model and try to make it better with by combining the garch results.
# 
# On the other hand, we should be aware of the mimick behaviours of the LSTM and in our case, our model try to mimicks the real data with a lag.
# 

# In[42]:


skor2 = rmse_tr(predicted_stock_price, np.array(df.loc[df.index[1000:], 'Volatility']))
print("Root Mean Squared Error of the model is calculated as ",skor2)


# ## Neural-Garch Model (Combining Garch(1,1) and LSTM)

# In[31]:


training_set = df.iloc[:, 10:12].values
# 100 timestep ve 1 çıktı ile data yapısı oluşturalım
X_train = []
y_train = []
for i in range(1000, df.shape[0]):X_train.append(training_set[i-1000:i,:])
for i in range(1000, df.shape[0]):y_train.append(training_set[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[32]: ###Remaining after this


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))     ##Did 1 instead of 2


# In[56]:


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True, input_shape = (X_train.shape[1], 2))) #,1 instead of ,2 in the end)
regressor.add(Dropout(0.1))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10))
regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# In[57]:


regressor.save('my_modelp2.h5')


# In[34]:


from keras.models import load_model
regressor = load_model('my_modelp2.h5')


# In[35]:


predicted_stock_price = regressor.predict(X_train)


# In[39]:


# Visualising the results
plt.figure(figsize=(18,6))
plt.plot(df.iloc[1000:, 10:11].values, color = 'red', label = 'Observed Volatility')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Volatility By LSTM-GARCH(1,1)')
plt.title('Real Rolling Volatility vs Forecast of LSTM-GARCH(1,1)')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()


# In[37]:


skor3 = rmse_tr(predicted_stock_price, np.array(df.loc[df.index[1000:], 'Volatility']))
print("Root Mean Squared Error of the model is calculated as ",skor3)


# In[ ]:


