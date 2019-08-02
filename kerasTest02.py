import os
import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import math
from sklearn.metrics import mean_squared_error
import io

# dataset 생성 함수 정의
look_back = 1
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
  
# csv 파일 불러오기
df = pd.read_csv('kospi200test_1.csv')

# print(df)
# print(df.shape)
# print(df[:10])

# '종가(close)' 열만 따로 빼내어 정의
dataset = df['close'].values[::-1]
dataset.astype('float32')
# print(dataset)
# print(dataset.shape)
# print(dataset[:10])

dataset = dataset.reshape(1,-1)
# print(dataset.shape)

# 데이터 정규화(normalization)
scaler = MinMaxScaler(feature_range=(0,1))
nptf = scaler.fit_transform(dataset)
# print(nptf)

# train, test로 각각 데이터 split
train_size= int(len(nptf) * 0.67)
test_size = len(nptf) - train_size
train, test = nptf[0:train_size,:], nptf[train_size:len(nptf),:]
# print(len(train), len(test))
# print(train)
# print("------여기까지 학습데이터")
# print(test)
# print(train.shape)
# print(test.shape)

# 학습을 위한 dataset 생성
look_back = 1
x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)
# print(x_train.shape)

# LSTM에 집어넣을 수 있도록 shape 조정
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))

# print(x_train.shape)
# print(x_test.shape)
'''
# LSTM을 이용한 모델링
model = Sequential()
model.add(LSTM(6, input_shape=((1, look_back))))
model.add(Dense(1))

model.summary()

# 학습
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

# predict 값 예측
testPredict = model.predict(x_test)
testPredict = scaler.inverse_transform(testPredict)
# y_test = scaler.inverse_transform(y_test)
testScore = math.sqrt(mean_squared_error(y_test, testPredict))
print('Train Score: %.2f RMSE' % testScore)
 
# 데이터 입력 마지막 다음날 종가 예측
lastX = nptf[-1]
lastX = np.reshape(lastX, (1, 1, 1))
lastY = model.predict(lastX)
lastY = scaler.inverse_transform(lastY)
print('Predict the Close value of final day: %.2f' % lastY)
'''
