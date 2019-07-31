from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# 책 127page
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape) 
# print(test_data)
# print(train_data)
# print("표준화 이전 데이터")

# mean = train_data.mean(axis=0)
# train_data -= mean
# std = train_data.std(axis=0)
# train_data /= std #데이터 표준화 standardization 
# print(train_data)
# test_data -= mean
# test_data /= std

scaler= StandardScaler() # StandardScaler 클래스 변수에 대입 @@@@@@@@@@얘랑 .fit은 한번만 해@@@@@@@@@@@@
# scaler = MinMaxScaler()
scaler.fit(train_data) # x 데이터를 standardscale 해라(연산할 데이터만! y는 하지 않는다.) 
# 한 데이터에 대해 fit을 했다면 다른 데이터에 대해서는 다시 할 필요가 없다!!

train_data = scaler.transform(train_data) # 모양을 변형해주면 끝! 좀 더 원활한 데이터 예측이 가능해진다.
# print(train_data_scaled)

test_data = scaler.transform(test_data)
# print(test_data_scaled)



from keras import models
from keras import layers

def build_model():
    # 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                            input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model # 함수로 모델을 만들 때 반드시 들어가야하는 모델 리턴!

from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5)
all_scores = []
for train_index, test_index in kf.split(train_data, train_targets):
    partial_train_data, val_data = train_data[train_index], train_data[test_index]
    partial_train_targets, val_targets = train_targets[train_index], train_targets[test_index]

'''
k = 5 # 4조각으로 나눠서 3조각은 train 1조각은 test =>1,2,3/4 _ 2,3,4/1 _ 1,3,4/2 _ 1,2,4/3
      # 4번을 수동으로 돌린 것과 같은 효과    
num_val_samples = len(train_data) // k

print(len(train_data))
print(num_val_samples)


all_scores = []
for i in range(k):
    print("처리중인 폴드 #", i)
    # 검증 데이터 준비 : k번째 분할
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i +1) * num_val_samples]

    # 훈련 데이터 준비: 다른 분할 전체
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]],
        axis=0)
'''
num_epochs = 1


# 케라스 모델 구성(컴파일 포함)
model = build_model()
# 모델 훈련(verbose=0 이므로 훈련 과정이 출력되지 않습니다.)
model.fit(partial_train_data, partial_train_targets,
            epochs=num_epochs, batch_size=1, verbose=0) # verbose = 0
# 검증 세트로 모델 평가
val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))

# metrics에서 mae 받아와서 evaluate에서 사용
