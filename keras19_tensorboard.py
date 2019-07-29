#1. 데이터
import numpy as np 

x = np.array([range(1000), range(3110,4110), range(1000)]) # 3행 100열 / dim=100 / shape(100,)
y = np.array([range(5010,6010)])

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
print(len(x_train), "train +",len(x_test),"test")
print(len(y_train), "train +",len(y_test),"test")
print(len(x_val), "x_val +",len(y_val),"y_val")


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping, TensorBoard

model = Sequential()

# model.add(Dense(5000, input_dim = 3, activation = 'relu'))
model.add(Dense(100, input_shape = (3, ), activation = 'relu',
                 kernel_regularizer = regularizers.l2(0.01))) # 통상 0.01
# model.add(Dense(1000, kernel_regularizer = regularizers.l2(0.01)))
# model.add(BatchNormalization())
model.add(Dense(100))
# model.add(Dropout(0.2)) # 노드 갯수의 20% 줄인다. 1000개->800개
model.add(Dense(100))
# model.add(Dropout(0.9)) # 노드를 삭제하는 것이 아니라 사용하지 않는 것
model.add(Dense(1)) # dim =>2나 3으로 바뀌어도 y값(output)에 따라 값 조절


model.summary()

#3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0,
                      write_graph=True, write_images=True)
# cmd창에서 graph폴더의 상위폴더로 dr 변경한후, tensorboard --logdir ./graph 실행 후 인터넷 접속

early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.fit(x_train, y_train, epochs=100, batch_size = 100, verbose=3,
          validation_data= (x_val, y_val), # 스스로 학습하는 동시에 검증하라.(훈련이 더 잘됌)
          callbacks=[early_stopping, tb_hist])  # tb_hist: 텐서보드 접근하는 callbacks 함수

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

# R2(결정계수) 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

print("loss : ", loss) # 0.001 이하로 만들기
