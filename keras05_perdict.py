#1. 데이터
import numpy as np 
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 10행 1열
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x3 = np.array([101,102,103,104,105,106]) # 6행 1열
x2 = np.array([51,52,53,54,55,56,57,58,59,60,61,62,63,64,65]) # 15행 1열
x4 = np.array(range(30,50))

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

# model.add(Dense(5, input_dim = 1, activation = 'relu'))
model.add(Dense(5, input_shape = (1, ), activation = 'relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

model.summary()


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size = 1)
# model.fit(x_train, y_train, epochs=100)

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)


y_predict = model.predict(x4)
print(y_predict)

model.summary()
