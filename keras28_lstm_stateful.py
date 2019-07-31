import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
a = np.array(range(1,101))
batch_size = 3
size = 5

def split_5(seq, size): # array의 데이터를 5개씩 잘라서 [1,2,3,4,5] 연속되는 숫자데이터 >LSTM
    aaa = []
    for i in range(len(a)-size + 1): # range(6) = 0~5 ==>>> 자른 갯수 + 1 = 행의 갯수
        subset = a[i:(i+size)]
        aaa.append(subset)
        #aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)  

dataset = split_5(a, size)
print("==========================")
print(dataset)
print(dataset.shape)

x_train = dataset[:,0:4]
y_train = dataset[:,4]

x_train = np.reshape(x_train, (len(x_train), size-1, 1))

x_test = x_train + 100 # 101~200
y_test = y_train + 100 # 105~200

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_test[0])


#2. 모델구성
model = Sequential() 
model.add(LSTM(128, batch_input_shape=(batch_size,4,1), # (batch_size값, 열, 자를갯수)
                 stateful=True)) # stateful => 상태 유지를 해라. 디폴트는 false겠지?
model.add(Dense(130))
model.add(Dense(100))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='min')
tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0,
                      write_graph=True, write_images=True)

num_epochs = 5

history_l = []
for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    history = model.fit(x_train, y_train, epochs=100, batch_size=batch_size,
                        verbose=2, shuffle=False, # 데이터를 섞지않고(초기화하지않고) 유지하겠다. epoch하나 끝나면 훈련상태를 그대로 다시 가져오겠다.
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping, tb_hist])
    model.reset_states() # 리셋했다고해서 지워지는건 아니다. 상태유지lstm에서는 꼭 넣어줘야함
    history_l.append(history)

print(history)

mse, _ = model.evaluate(x_train, y_train, batch_size=batch_size)
print("mse : ", mse)
model.reset_states() # 여기서도 리셋한번 더. 상태값이 변하는 것은 아님!!! 위에서 shuffle과 stateful이라는 안전장치를 설정해두었기 때문.

y_predict = model.predict(x_test, batch_size=batch_size)

print(y_predict[0:5])

# RMSE
def RMSE(y_test, y_predict): # y_test, y_predict의 차이를 비교하기 위한 함수
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt 제곱근씌우기
print("RMSE : ", RMSE(y_test, y_predict))

# R2 결정계수
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# 히스토리에 있는 모든 데이터 나열
# print(history_l.history.keys())  # 아래 히스토리 요약에 key값을 여기서 나오는애들로 넣어줘야됨!!

# matplotlib을 이용한 데이터 시각화
import matplotlib.pyplot as plt

# mse에 대한 히스토리 요약
for i in range(len(history_l)):
   plt.plot(history_l[i].history['mean_squared_error'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper right') # for문 돌때마다 발생하는 값들을 리스트로 넣는방법 찾기.
plt.show()
