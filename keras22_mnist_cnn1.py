#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt 
import numpy 
import os 
import tensorflow as tf

# 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28 ,28 ,1).astype('float32') / 255 # 6만행(무시) 나머지는 아래 input_shape값이 된다.
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255 # 0~1 사이로 수렴시키기 위해 minmaxscaler같은거 필요없이 각 픽셀당 255의 값을 나누어서 데이터 전처리를 하는 과정
Y_train = np_utils.to_categorical(Y_train) # One Hot Incoding으로 데이터를 변환시킨다. 분류
Y_test = np_utils.to_categorical(Y_test)

print(X_train.shape)
print(X_test.shape)


# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # CNN 분류모델에서는 마지막 activation은 분류모델이라는 것을 인식시켜주기 위해서 softmax여야한다. # 0~9까지 10개의 데이터를 보내줄게

model.compile(loss='categorical_crossentropy', # 분류모델에선 loss='mse' 대신 이걸 쓴다!
              optimizer='adam',
              metrics=['accuracy'])

# 모델 최적화 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

#모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                     epochs=30, batch_size=200, verbose=1,
                     callbacks=[early_stopping_callback])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
