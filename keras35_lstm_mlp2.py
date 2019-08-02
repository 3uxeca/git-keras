import numpy as np 
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Flatten

a = np.array(range(1,101))


size = 8
def split_8(seq, size): # array의 데이터를 5개씩 잘라서 [1,2,3,4,5,6,7,8]
    aaa = []
    for i in range(len(a)-size + 1): # range(6) = 0~5 ==>>> 자른 갯수 + 1 = 행의 갯수
        subset = a[i:(i+size)]
        aaa.append(subset)
        #aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa) 

dataset = split_8(a, size)
print("====================")
# print(dataset)

x_train = dataset[:,0:4] # 93행 4열 만들기
y_train = dataset[:,4:8,] # 93행 4열 만들기
print(x_train.shape)        # (93, 4)
print(y_train.shape)        # (93, 4) reshape필요
print(x_train[0:2,])
print(y_train[0:2,])

# x_train = np.reshape(x_train, (6,4,1))
x_train = np.reshape(x_train, (len(a)-size+1,2,2))
# y_train = np.reshape(y_train, (len(a)-size+1,4,1))

print(x_train.shape)    #(93,4,1)
print(y_train.shape)

# x_test = x_train + 110
# y_test = y_train + 110
# x_test = np.array([[[11],[12],[13],[14]], [[12],[13],[14],[15]],
#                     [[13],[14],[15],[16]], [[14],[15],[16],[17]]])
                    # 가장 작은 괄호의 수=1, 중간 괄호의 수=4, 제일 큰 괄호의 수=4 => (4, 4, 1) 

# y_test = np.array([105, 106, 107, 108])

from sklearn.model_selection import train_test_split # 사이킷런의 분할기능(행에맞춰분할)
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, random_state=66, test_size=0.4 # test를 40%로 train을 60%의 양으로 분할
)
x_train, x_val, y_train, y_val = train_test_split( # train 60 val 20 test 20 으로 분할
    x_train, y_train, random_state=66, test_size=0.5 # train은 위에서 나눴고, 
 )                                                # 40으로 나누어진 test를 다시 반으로 분할


print(x_test.shape)     #(4,4,1)
print(y_test.shape)     #(4, )


#2. 모델 구성
model = Sequential()

model.add(LSTM(10, input_shape=(2,2), return_sequences=True))
model.add(LSTM(10, return_sequences=True)) # return_sequences의 역할: 두 LSTM을 연결하는 다리
model.add(LSTM(10, return_sequences=True)) # return_sequences의 역할: 두 LSTM을 연결하는 다리
model.add(LSTM(10)) # output값을 보여줘야하므로 Dense층과 연결 필요

model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(4))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
          callbacks=[early_stopping]  )

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # y_test, y_predict의 차이를 비교하기 위한 함수
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt 제곱근씌우기
print("RMSE : ", RMSE(y_test, y_predict))

# R2(결정계수) 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)