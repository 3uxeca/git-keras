# 모델구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping, TensorBoard

model = Sequential()

# model.add(Dense(5000, input_dim = 3, activation = 'relu'))
model.add(Dense(100, input_shape = (3, ), activation = 'relu'))
                 #,kernel_regularizer = regularizers.l2(0.01))) # 통상 0.01
# model.add(Dense(1000, kernel_regularizer = regularizers.l2(0.01)))
# model.add(BatchNormalization())
model.add(Dense(100))
# model.add(Dropout(0.2)) # 노드 갯수의 20% 줄인다. 1000개->800개
model.add(Dense(100))
# model.add(Dropout(0.9)) # 노드를 삭제하는 것이 아니라 사용하지 않는 것
model.add(Dense(1)) # dim =>2나 3으로 바뀌어도 y값(output)에 따라 값 조절


#model.summary()

model.save('savetest01.h5')
print("저장 잘 됐당!")