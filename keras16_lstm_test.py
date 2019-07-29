import numpy as np 
from keras.models import Sequential 
from keras.layers import Dense, LSTM

a = np.array(range(11,21))


size = 5
def split_5(seq, size): # array의 데이터를 5개씩 잘라서 [1,2,3,4,5]
    aaa = []
    for i in range(len(a)-size + 1): # range(6) = 0~5 ==>>> 자른 갯수 + 1 = 행의 갯수
        subset = a[i:(i+size)]
        aaa.append(subset)
        #aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa) 

dataset = split_5(a, size)
print("====================")
print(dataset)

x_train = dataset[:,0:4] # 6행 4열 만들기
y_train = dataset[:,4,] # 6열 만들기
print(x_train.shape)        # (6, 4)
print(y_train.shape)        # (6, ) reshape필요

# x_train = np.reshape(x_train, (6,4,1))
x_train = np.reshape(x_train, (len(a)-size+1,4,1))

print(x_train.shape)    #(6,4,1)

x_test = np.array([[[11],[12],[13],[14]], [[12],[13],[14],[15]],
                    [[13],[14],[15],[16]], [[14],[15],[16],[17]]])
                    # 가장 작은 괄호의 수=1, 중간 괄호의 수=4, 제일 큰 괄호의 수=4 => (4, 4, 1) 
y_test = np.array([15, 16, 17, 18])

print(x_test.shape)     #(4,4,1)
print(y_test.shape)     #(4, )

#2. 모델 구성
model = Sequential()

model.add(LSTM(10, input_shape=(4,1), return_sequences=True))
# model.add(LSTM(10, return_sequences=True)) # return_sequences의 역할: 두 LSTM을 연결하는 다리
# model.add(LSTM(10, return_sequences=True)) # return_sequences의 역할: 두 LSTM을 연결하는 다리
# model.add(LSTM(10, return_sequences=True)) # return_sequences의 역할: 두 LSTM을 연결하는 다리
model.add(LSTM(10)) # output값을 보여줘야하므로 Dense층과 연결 필요

model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train, y_train, epochs=300, batch_size=300, verbose=3,
          callbacks=[early_stopping]  )

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)
