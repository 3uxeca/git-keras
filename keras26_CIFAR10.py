from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import regularizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt 

# CIFAR_10은 3채널로 구성된 32x32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3 # input.shape = (32,32,3)
IMG_ROWS = 32
IMG_COLS = 32

## 상수 정의
# BATCH_SIZE = 128
BATCH_SIZE = 800
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 2
# VERBOSE = 1
VALIDATION_SPLIT = 0.2 # 5만개의 데이터중 4만개의 학습데이터와 1만개의 검증데이터로 나누어짐
OPTIM = RMSprop()

## 데이터셋 불러오기
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
print(X_train.shape, 'train samples')
# print(X_test.shape[0], 'test samples')
print(X_test.shape, 'test samples')

## 범주형으로 변환 (기계가 빨리 번역해서 인식하게 하기 위해서, one hot encoding!!)
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

## 실수형으로 지정하고 정규화
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32') # 255로 나누었을때 0과 1사이의 실수값으로 나타내야하기 때문에.
# X_train /= 255 # minmax 또는 standard scale 사용 가능
# X_test /= 255 # 한쪽으로 몰려있는 데이터라면 정규화가 어려웠을 것.

### 여기 MINMAX, STANDARD SCALER 밑에 각각 적용해서 비교해보기#####
x_train0 = X_train.shape[0]
x_test0 = X_test.shape[0]

X_train_scale = X_train.reshape([x_train0,IMG_ROWS*IMG_COLS*IMG_CHANNELS])
X_test_scale = X_test.reshape([x_test0,IMG_ROWS*IMG_COLS*IMG_CHANNELS])
print(X_train_scale.shape)
print(X_test_scale.shape)

scaler = StandardScaler() # StandardScaler 클래스 변수에 대입
# scaler = MinMaxScaler()
scaler.fit(X_train_scale)
X_train_scaled = scaler.transform(X_train_scale)
X_test_scaled = scaler.transform(X_test_scale)
print(X_train_scaled)
print(X_test_scaled)

X_train = X_train_scaled.reshape([x_train0,IMG_ROWS,IMG_COLS,IMG_CHANNELS])
X_test = X_test_scaled.reshape([x_test0,IMG_ROWS,IMG_COLS,IMG_CHANNELS])
print(X_train.shape)
print(X_test.shape)
# print(X_train.shape)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

# 사진 한 장 뽑기!
# pic = X_train[44] # 6만가지의 이미지 중 원하는 n번째 이미지를 보여줘라.
# plt.imshow(pic, cmap=plt.cm.binary)
# plt.show()


## 신경망 정의
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.01),
                   input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu')) 
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))) #
model.add(Activation('relu'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2,2)))#
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))) #
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))) #
model.add(Activation('relu'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2,2)))#
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))) #
model.add(Activation('relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))) #
model.add(Activation('relu'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2,2)))#
model.add(Dropout(0.4))

model.add(Flatten()) # 이하 DNN\
model.add(Dense(NB_CLASSES)) # NB_CLASSES = 10 = Output 분류모델에서는 무조건 주어진 10개 중 선택해야한다.
model.add(Activation('softmax'))

model.summary()

## 학습
model.compile(loss='categorical_crossentropy', optimizer=OPTIM,
               metrics=['accuracy'])

tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0,
                      write_graph=True, write_images=True)

# 모델 최적화 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30)

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,
                    verbose=VERBOSE,
                    callbacks=[early_stopping_callback, tb_hist])

print('Testing...')
score = model.evaluate(X_test, Y_test,
                       batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score:", score[0]) # loss
print('Test acuuracy:', score[1]) # acc


# 히스토리에 있는 모든 데이터 나열
print(history.history.keys())

'''
## 시각화 작업

# 단순 정확도에 대한 히스토리 요약
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 손실에 대한 히스토리 요약
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# 정확도&손실 히스토리 종합 요약
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model loss, accuracy')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss', 'train acc', 'test acc']) #위 history순서와 대응
plt.show()
'''