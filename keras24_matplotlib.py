#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy 
import os 
import tensorflow as tf

# 데이터 불러오기


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# import matplotlib.pyplot as plt 
# digit = X_train[44] # 6만가지의 이미지 중 원하는 n번째 이미지를 보여줘라.
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show() # jupyter notebook이나 구글 colab 환경에서는 plt.show() 코드를 쓰지 않아도 이미지가 출력된다.

X_train = X_train.reshape(X_train.shape[0], 28 ,28 ,1).astype('float32') / 255 # 6만행(무시) 나머지는 아래 input_shape값이 된다.
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255 # 0~1 사이로 수렴(minmax)시키기 위해 minmaxscaler같은거 필요없이 각 픽셀당 255의 값을 나누어서 데이터 전처리를 하는 과정
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
                     epochs=5, batch_size=200, verbose=1,
                     callbacks=[early_stopping_callback])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1])) # 분류모델에서는 accuracy가 정확하다.(회귀모델에서는 mse나 R2를 사용했었음.)

print(history.history.keys()) # 위의 history라는 변수의 반환값 확인 (['val_loss', 'val_acc', 'loss', 'acc'])

#colab GPU환경에서 구동
import matplotlib.pyplot as plt


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model loss, accuracy')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'], loc='lower left') #위 history순서와 대응
plt.show()