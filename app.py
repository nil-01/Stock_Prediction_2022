import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import  streamlit as st
from datetime import date
import webbrowser



url='https://stockanalysis.com/stocks/'
start='1990-1-1'
end=date.today().strftime("%Y-%m-%d")
st.title('Stock Prediction')
user_input=st.text_input('Enter Stock ticker','AAPL')

df=data.DataReader(user_input,'yahoo',start,end)


# Data Describing
st.subheader('If You Don{}t know the Stock ticker Click Below Button'.format("'"))



#adding a button

if st.button('TICKER'):
    webbrowser.open(url)
st.subheader('Data from 1990 - Current Year')
st.write(df.describe())
#print('If You Don{}t know the Stock ticker Press 1'.format(','))
#help_input


#visualzing

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with ma100')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with ma100 with ma200')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma200,'g')
plt.plot(ma100,'r')
st.pyplot(fig)

# splitting data into training and testing

data_train = pd.DataFrame(df['Close'][0:int(len(df) * 0.80)])
data_test = pd.DataFrame(df['Close'][int(len(df) * 0.80):int(len(df))])
print(data_train.shape)
print(data_test.shape)

 #scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array=scaler.fit_transform(data_train)




# load model

model=load_model('Stackapp.h5')

#test

past_100_days= data_train.tail(100)

final_df=past_100_days.append(data_test,ignore_index=True)

actual_data=scaler.fit_transform(final_df)

#split

x_test = []
y_test = []

for i in range(100, actual_data.shape[0]):
    x_test.append(actual_data[i - 100:i])
    y_test.append(actual_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

## Making prediction
y_predict =model.predict(x_test)
scaler= scaler.scale_

scale_factor=1/scaler[0]
y_predict = scale_factor*y_predict
y_test=scale_factor*y_test

#final

st.subheader("Predicted Price Vs Original Price")

fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label='Original Price')
plt.plot(y_predict,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader("Dedicated To Angel")
