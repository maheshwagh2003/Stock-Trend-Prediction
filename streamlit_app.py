import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from pandas_datareader import data as pdr
import yfinance as yf
import datetime
from keras.models import load_model
import streamlit as st


# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
start = '2010-01-01'

x = datetime.datetime.now()
date = x.strftime("%Y-%m-%d")


st.title('Stock Trend Prediction')
st.subheader('-Mahesh Wagh')

user_input = st.text_input('Enter Stock Ticker', 'TSLA')


yf.pdr_override()
data = pdr.get_data_yahoo(user_input, start = '2010-01-01', end = date)

df = data.reset_index()
#df = df.drop(['Date', 'Adj Close'], axis = 1)


#Describing Data
st.subheader('Data From 2010 - 2023')
st.write(df.describe())


#Visualizations

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 200 Moving Average')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 and 200 Moving Average')
ma200 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r', label='Moving Avg 100 days')
plt.plot(ma200, 'g', label='Moving Avg 200 days')
plt.plot(df.Close, 'b', label='Closing Price')
plt.legend()
st.pyplot(fig)

#Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#Loading Model

model = load_model('keras_model.keras')


#Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat((past_100_days, data_testing), axis=0, ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])



x_test, y_test = np.array(x_test), np.array(y_test)


#Making Predictions

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test * scale_factor

#Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)





