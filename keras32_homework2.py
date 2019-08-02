from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Conv2D, Flatten
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

#데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
'''
import matplotlib.pyplot as plt # 이미지 불러오는 부분
digit = X_train[5900]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
'''

X_train =  X_train.reshape(X_train.shape[0], 28 , 28, 1).astype('float32') / 255
X_test =  X_test.reshape(X_test.shape[0], 28 , 28, 1).astype('float32') / 255

print(Y_train.shape)
print(Y_test.shape)


Y_train = np_utils.to_categorical(Y_train) # OneHot Encording 0 ~ 9로 분류
Y_test = np_utils.to_categorical(Y_test)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


def build_network(keep_prob = 0.5, optimizer='adam'):
    inputs = Input(shape=(28, 28, 1), name = 'input')
    x = Conv2D(2, (3,3))(inputs)
    x = Dropout(keep_prob)(x)
    x = Conv2D(2, (3,3))(x)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    prediction = Dense(10, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = prediction)
    model.compile(optimizer = 'optimizer', loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn = build_network, verbose = 1)

# hyperparameters = create_hyperparameters()

# from sklearn.model_selection import RandomizedSearchCV
# search = RandomizedSearchCV(estimator = model, param_distributions = hyperparameters, n_iter= 10, n_jobs= 1, cv= 3, verbose=1)

# search.fit(X_train, Y_train)

# print(search.best_params_)
