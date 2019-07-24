from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # 4행 3열
y = array([4,5,6,7]) # 1행 4열

print("x.shape: ", x.shape)
print("y.shape: ", y.shape) # (4,) 결과값의 갯수: 4개

x = x.reshape((x.shape[0], x.shape[1], 1)) #LSTM에 넣기 위한 모양 작업(데이터갯수변함x)

print("x.shape: ", x.shape)

#2. 모델 구성  # LSTM의 기본적인 모양 input_shape(/행무시/열,몇개씩자를거야)
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3,1))) # 행 무시, dim = 3, 1개씩 잘라서 수행
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

#3. 실행
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x, y, epochs=100)

x_input = array([6,7,8]) # 1행, 3열, 몇개씩자를건데??
x_input = x_input.reshape((1,3,1)) # 3열 1개씩

yhat = model.predict(x_input)
print(yhat)