#1. 데이터
import numpy as np 
x_train = np.arange(1,101,1)
y_train = np.arange(501,601,1)
x_test = np.arange(1001,1101,1)
y_test = np.arange(1101,1201,1)


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(30, input_dim = 1, activation = 'relu'))
model.add(Dense(29))
model.add(Dense(29))
model.add(Dense(29))
model.add(Dense(29))
model.add(Dense(29))
model.add(Dense(29))
model.add(Dense(29))
model.add(Dense(29))
model.add(Dense(1))

# model.summary()


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=111, batch_size = 9)
# model.fit(x_train, y_train, epochs=100)

#4. 평가 예측
loss, acc = model.evaluate(x_train, y_train, batch_size=1)
print("acc : ", acc)


y_predict = model.predict(y_test)
print(y_predict)


model.summary()