import numpy as np 
from keras.models import Sequential 
from keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler

a = np.array(range(1,11))
# a = np.array(range(11,21))


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
print(a)

x_train = dataset[:,0:4] # 6행 4열 만들기 [:,0:4]의 의미 =  앞의':'는 전체행, 0:4는 열을 4크기로 자르는것
print(x_train.shape)        # (6, 4)
print(x_train)
y_train = dataset[:,4] # 6열 만들기. dataset의 4번째 열 전체 값을 리스트로 가져오기
print(y_train.shape)        # (6, ) reshape필요
print(y_train)



scaler = StandardScaler() # StandardScaler 클래스 변수에 대입 @@@@@@@@@@얘랑 .fit은 한번만 해@@@@@@@@@@@@
# scaler = MinMaxScaler()
scaler.fit(x_train) # x 데이터를 standardscale 해라(연산할 데이터만! y는 하지 않는다.) 
# 한 데이터에 대해 fit을 했다면 다른 데이터에 대해서는 다시 할 필요가 없다!!
x_train_scaled = scaler.transform(x_train) # 모양을 변형해주면 끝! 좀 더 원활한 데이터 예측이 가능해진다.
print(x_train_scaled)


# x_train = np.reshape(x_train, (6,4,1))
x_train_scaled = np.reshape(x_train_scaled, (len(a)-size+1,4,1))

print(x_train_scaled.shape)    #(6,4,1)

# x_test = np.array([[[11],[12],[13],[14]], [[12],[13],[14],[15]],
#                     [[13],[14],[15],[16]], [[14],[15],[16],[17]]])
#                     # 가장 작은 괄호의 수=1, 중간 괄호의 수=4, 제일 큰 괄호의 수=4 => (4, 4, 1) 
x_test = np.array([[11,12,13,14],[12,13,14,15],[13,14,15,16],[14,15,16,17]])
y_test = np.array([15, 16, 17, 18])


# scaler = StandardScaler() # StandardScaler 클래스 변수에 대입
# scaler = MinMaxScaler()
# scaler.fit(x_test) # x 데이터를 standardscale 해라(연산할 데이터만! y는 하지 않는다.) 
# 한 데이터에 대해 fit을 했다면 다른 데이터에 대해서는 다시 할 필요가 없다!!
x_test_scaled = scaler.transform(x_test) # 모양을 변형해주면 끝! 좀 더 원활한 데이터 예측이 가능해진다.

x_test_scaled = x_test_scaled.reshape(x_test_scaled.shape[0],x_test_scaled.shape[1],1)
print(x_test_scaled.shape)     #(4,4,1)
print(y_test.shape)     #(4, )
print(x_test_scaled)

#2. 모델 구성
model = Sequential()

model.add(LSTM(32, input_shape=(4,1), return_sequences=True))
# model.add(LSTM(10, return_sequences=True)) # return_sequences의 역할: 두 LSTM을 연결하는 다리
model.add(LSTM(10)) # output값을 보여줘야하므로 Dense층과 연결 필요

model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(8))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train_scaled, y_train, epochs=3000, verbose=3,
          callbacks=[early_stopping]  )

#4. 평가 예측
loss, acc = model.evaluate(x_test_scaled, y_test)

y_predict = model.predict(x_test_scaled)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)
