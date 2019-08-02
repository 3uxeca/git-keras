from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers

import numpy 
import os 
import tensorflow as tf

# 데이터 불러오기


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28 ,28 ,1).astype('float32') / 255 # 6만행(무시) 나머지는 아래 input_shape값이 된다.
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255 # 0~1 사이로 수렴(minmax)시키기 위해 minmaxscaler같은거 필요없이 각 픽셀당 255의 값을 나누어서 아주 간단하게 데이터 전처리를 하는 과정
Y_train = np_utils.to_categorical(Y_train) # One Hot Incoding으로 데이터를 변환시킨다. 분류
Y_test = np_utils.to_categorical(Y_test)

X_train = X_train[0:300]
Y_train = Y_train[0:300]
print(X_train.shape)
print(X_test.shape)



# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
# model.add(Dense(60, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # CNN 분류모델에서는 마지막 activation은 분류모델이라는 것을 인식시켜주기 위해서 softmax여야한다. # 0~9까지 10개의 데이터를 보내줄게

model.compile(loss='categorical_crossentropy', # 분류모델에선 loss='mse' 대신 이걸 쓴다!
              optimizer='adam',
              metrics=['accuracy'])

# 이미지 생성기 ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(
    rotation_range=20, # 회전값 범위 20
    width_shift_range=0.02, # 넓이 0.02
    height_shift_range=0.02, # 높이 0.02
    horizontal_flip=True # 수평값
)
# model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=32), # 훈련을 시키되, 발전기도 같이 실행해라!
#                     steps_per_epoch=len(X_train) // 32, # 몇 배로 증폭시킬건지 steps*batch_size 자기 숫자만큼의 새로운 이미지 데이터 생성
#                     epochs=200,
#                     validation_data=(X_test, Y_test),
#                     verbose=1 #, callbacks=callbacks
#                     ) 


# 모델 최적화 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30)

#모델의 실행
history = model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=600), # 훈련을 시키되, 발전기도 같이 실행해라!
                    steps_per_epoch=120, # 몇 배로 증폭시킬건지 steps*batch_size 자기 숫자만큼의 새로운 이미지 데이터 생성
                    epochs=50,
                    validation_data=(X_test, Y_test),
                    verbose=1 #, callbacks=callbacks
                    ) # 그냥 model.fit 보다 작업 처리 속도가 느리다. 증폭시킨 후 돌리는 것이기 때문에

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
# 분류모델에서는 accuracy가 정확하다.(회귀모델에서는 mse나 R2를 사용했었음.)
