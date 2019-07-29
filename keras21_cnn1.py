from keras.models import Sequential
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# filter_size = 32 # output값 (3,3)으로 자른 이미지를 32장 만들어내라.
filter_size = 7 # (2x2)로 잘라서 나온게 4x4개. 이것을 7장을 만들어내라.
# kernel_size = (3,3) # 가로3 세로3으로 잘라서 특성을 파악
kernel_size = (2,2)

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
model = Sequential()
model.add(Conv2D(filter_size, kernel_size,  # filter_size : output값, kernel_size : 몇개씩 자를거냐(가로세로3씩자를거임)
                  padding = 'same', input_shape = (5,5,1)))
                  # padding:경계처리방법. 유효영역만 출력. 출력이미지사이즈<입력사이즈. 디폴트값='valid', 'same'은 출력=입력
                  # input_shape : 가로세로 28에 1은 흑백인 이미지를 입력값으로 넣겠다. 
                  # input_shape = (28,28,1))) # 컬러 이미지의 경우 3. 앞으로 28,28 크기의 입력만 하라.
# model.add(Conv2D(16,(2,2)))
model.add(MaxPooling2D(2,2)) # 이미지를 2x2 크기로 나눠 각각의 최대값들을 다시 2x2 크기로 반환
# model.add(Conv2D(8,(2,2)))
model.add(Flatten()) # CNN의 마지막에 Output layer인 Dense층과 연결해주기 위해 들어간다.
model.add(Dense(2))


model.summary()