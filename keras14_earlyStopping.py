#1. 데이터
import numpy as np 

# x = np.array(range(1,101))
# y = np.array(range(1,101))
x = np.array([range(100), range(311,411), range(100)]) # 100행 3열 / dim=3 / shape(3,)
y = np.array([range(501,601), range(711, 811), range(100)])

print(x.shape)
print(y.shape)

#ValueError: Error when checking input: expected dense_1_input to have shape (3,) but got array with shape (100,) 행렬전치해라.
x = np.transpose(x)
y = np.transpose(y)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split # 사이킷런의 분할기능(행에맞춰분할)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.4 # test를 40%로 train을 60%의 양으로 분할
)
x_val, x_test, y_val, y_test = train_test_split( # train 60 val 20 test 20 으로 분할
    x_test, y_test, random_state=66, test_size=0.5 # train은 위에서 나눴고, 
 )                                                # 40으로 나누어진 test를 다시 반으로 분할\

print(x_test.shape)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense 
from keras.callbacks import EarlyStopping
model = Sequential()

model.add(Dense(10, input_dim = 3, activation = 'relu'))
# model.add(Dense(151, input_shape = (2, ), activation = 'relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3)) # dim =>2로 바뀌면서 output도 2로 변경


# model.summary()

# monitor = 'loss'
# 'val_loss'   # 'val_acc'  >>> 얘네 쓰려면 꼭 model.fit에 validaion_data 필요

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor='val_acc', patience=30, mode='auto')
model.fit(x, y, epochs=10000, verbose=1, validation_data=(x_val,y_val),
          callbacks=[early_stopping]) 

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # y_test, y_predict의 차이를 비교하기 위한 함수
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt 제곱근씌우기
print("RMSE : ", RMSE(y_test, y_predict))

# # MAE
# from sklearn.metrics import mean_absolute_error
# def MAE(y_test, y_predict): # y_test, y_predict의 차이를 비교하기 위한 함수
#     return mean_absolute_error(y_test, y_predict) 
# print("MAE : ", RMSE(y_test, y_predict))

# R2(결정계수) 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)