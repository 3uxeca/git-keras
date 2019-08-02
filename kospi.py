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
  
# csv 파일 불러오기
df = pd.read_csv('kospi200test_1.csv')

# print(df)
# print(df.shape)
# print(df[:10])

# '종가(close)' 열만 따로 빼내어 정의
dataset = df['close']
dataset.astype('float32')
# print(dataset)
print(dataset.shape)
# print(dataset[:10])

close_price= np.array(dataset)
# print(close_price)
print(close_price.shape)
close_price = close_price.transpose()
# print(dataset.shape)
# print(dataset)

size = 6
def split_6(seq, size): # array의 데이터를 5개씩 잘라서 [1,2,3,4,5,6,7,8]
    aaa = []
    for i in range(len(close_price)-size + 1): # range(6) = 0~5 ==>>> 자른 갯수 + 1 = 행의 갯수
        subset = close_price[i:(i+size)]
        aaa.append(subset)
        #aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa) 

data = split_6(close_price, size)
print("====================")
# print(dataset)

x_train = data[:,0:2] # 93행 4열 만들기
y_train = data[:,4:8,] # 93행 4열 만들기
print(x_train.shape)        # (93, 4)
print(y_train.shape)        # (93, 4) reshape필요
print(x_train[0:2,])
print(y_train[0:2,])

# # 데이터 정규화(normalization)
# scaler = MinMaxScaler(feature_range=(0,1))
# nptf = scaler.fit_transform(close_price)
# print(nptf)

from sklearn.model_selection import train_test_split # 사이킷런의 분할기능(행에맞춰분할)
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, random_state=66, test_size=0.4 # test를 40%로 train을 60%의 양으로 분할
)
x_test, x_val, y_test, y_val = train_test_split( # train 60 val 20 test 20 으로 분할
    x_test, y_test, random_state=66, test_size=0.5 # train은 위에서 나눴고, 
 )                                                # 40으로 나누어진 test를 다시 반으로 분할
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
# train, test로 각각 데이터 split
# train_size= int(len(nptf) * 0.67)
# test_size = len(nptf) - train_size
# train, test = nptf[0:train_size,:], nptf[train_size:len(nptf),:]
# print(len(train), len(test))
# print(train)
# print("------여기까지 학습데이터")
# print(test)
# print(train.shape)
# print(test.shape)


'''
# 학습을 위한 dataset 생성
look_back = 2
x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)
print(x_train)
print(x_train.shape)
'''

# # LSTM에 집어넣을 수 있도록 shape 조정
# x_train = np.reshape(x_train, (x_train.shape[0], 2, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], 2, x_train.shape[1]))
# # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))
# # x_test = np.reshape(x_test, (x_test.shape[0], x_train.shape[1], 2))

# print(x_train.shape)
# print(x_test.shape)


'''
# LSTM을 이용한 모델링
model = Sequential()
model.add(LSTM(6, input_shape=((2, 2))))
model.add(Dense(2))

model.summary()

# 학습
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2)

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
