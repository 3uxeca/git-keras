#1. 데이터 (앙상블.ensemble)
import numpy as np 


x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501,601), range(711, 811), range(100)])
x2 = np.array([range(100,200), range(311,411), range(100,200)])
y2 = np.array([range(501,601), range(711, 811), range(100)])

#ValueError: Error when checking input: expected dense_1_input to have shape (3,) but got array with shape (100,) 행렬전치해라.
x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split # 사이킷런의 분할기능(행에맞춰분할)
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=66, test_size=0.4 # test를 40%로 train을 60%의 양으로 분할
)
x1_val, x1_test, y1_val, y1_test = train_test_split( # train 60 val 20 test 20 으로 분할
    x1_test, y1_test, random_state=66, test_size=0.5 # train은 위에서 나눴고, 
 )                                                # 40으로 나누어진 test를 다시 반으로 분할\
## 파일이 2개니까 분할도 2번!!!!!!!!!!!!!
from sklearn.model_selection import train_test_split # 사이킷런의 분할기능(행에맞춰분할)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=66, test_size=0.4 # test를 40%로 train을 60%의 양으로 분할
)
x2_val, x2_test, y2_val, y2_test = train_test_split( # train 60 val 20 test 20 으로 분할
    x2_test, y2_test, random_state=66, test_size=0.5 # train은 위에서 나눴고, 
 )                                                # 40으로 나누어진 test를 다시 반으로 분할\
 
print(x2_test.shape)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input 

# 함수형 모델 Input
input1 = Input(shape=(3,)) # 첫번째 모델의 input
dense1 = Dense(100, activation='relu')(input1) # 앞의 100은 output
dense1_2 = Dense(30)(dense1)
dense1_3 = Dense(7)(dense1_2)

input2 =  Input(shape=(3,)) # 두번째 모델의 input
dense2 = Dense(50, activation='relu')(input2)
dense2_2 = Dense(7)(dense2)

# 두 모델 합치기!!! concatenate함수!!!! # 합친 두 모델을 merge라는 변수로 다시 구성하기
from keras.layers.merge import concatenate 
merge1 = concatenate([dense1_3, dense2_2]) # 합칠 때는 리스트 형태로

middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(7)(middle2) # 총 3개의 모델이 합쳐진 구성 확인

############################## 요기부터 아웃풋 모델

output1 = Dense(30)(middle3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(20)(middle3)
output2_2 = Dense(70)(output2)
output2_3 = Dense(3)(output2_2) # 총 5개의 모델 완성

model = Model(inputs = [input1, input2], 
              outputs = [output1_3, output2_3])

model.summary()



#3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10, batch_size = 1,
          validation_data= ([x1_val, x2_val], [y1_val, y2_val])) # 스스로 학습하는 동시에 검증하라.(훈련이 더 잘됌)
# model.fit(x_train, y_train, epochs=100)

#4. 평가 예측
# acc1 = list(model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1))
_, _, _, acc1, acc2 = (model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1))
print("acc1 : ", acc1)
print("acc2 : ", acc2)
# acc = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
# print("acc : ", acc)

y1_predict, y2_predict = model.predict([x1_test, x2_test])
print(y1_predict, y2_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
# def RMSE(y1_test, y2_test, y1_predict, y2_predict): # y_test, y_predict의 차이를 비교하기 위한 함수
#     return np.sqrt(mean_squared_error(y1_test, y1_predict)), np.sqrt(mean_squared_error(y2_test, y2_predict)) # np.sqrt 제곱근씌우기
# print("RMSE : ", RMSE(y1_test, y2_test, y1_predict, y2_predict))
def RMSE(xxx, yyy):
    return np.sqrt(mean_squared_error(xxx, yyy))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE: ", (RMSE1+RMSE2)/2) # 2개를 구했으니 전체 평균 값을 내는 것

# R2(결정계수) 구하기
from sklearn.metrics import r2_score # 외부에서 가져온 함수는 요구하는 변수가 정해져있으므로 주의
# r2_y1_predict, r2_y2_predict = r2_score(y1_test, y1_predict), r2_score(y2_test, y2_predict)
# print("R2 : ", r2_y1_predict, r2_y2_predict)
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
print("R2_1 : ", r2_y1_predict)
print("R2_2 : ", r2_y2_predict)
print("R2 : ", (r2_y1_predict + r2_y2_predict)/2)