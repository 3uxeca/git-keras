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

# from sklearn.model_selection import StratifiedKFold
seed=77
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor # wrappers 우리가 만든 keras모델을 sklearn 형태로 감쌀 것이다.
from sklearn.model_selection import KFold, cross_val_score # c_v_s 점수내는거
model = KerasRegressor(build_fn=build_model, epochs=10, # 이 모델은 회귀모델이다.
                        batch_size=1, verbose=1) # build_fn:어떤모델을 쓸거냐. 위에 만든 모델 쓰겠다.
kfold = KFold(n_splits=5, shuffle=True, random_state=seed) # 5번 따로 작업이기 때문에 유지가 필요없으므로 shuffle, 난수 77번째꺼 씀. 지정안하면 정말 무작위가 됨
results = cross_val_score(model, train_data, train_targets, cv=kfold) # model.fit()

import numpy as np 
print(results)
print(np.mean(results))