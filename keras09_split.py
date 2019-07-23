#1. 데이터
import numpy as np 

x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

x_train = x[:60]
y_train = y[:60]
x_val = x[60:80]
y_val = y[60:80]
x_test = x[80:]
y_test = y[80:]



#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5, input_dim = 1, activation = 'relu'))
# model.add(Dense(151, input_shape = (1, ), activation = 'relu'))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))


model.summary()

#3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs=100, batch_size = 1,
          validation_data= (x_val, y_val)) # 스스로 학습하는 동시에 검증하라.(훈련이 더 잘됌)
# model.fit(x_train, y_train, epochs=100)

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