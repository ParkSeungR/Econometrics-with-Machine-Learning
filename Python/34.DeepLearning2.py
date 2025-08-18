#!/usr/bin/env python
# coding: utf-8

# # 3.  케라스 모형의 저장과 로드

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:/JupyterWDirectory/MyStock")
# 현재 작업공간(working directory)확인  
os.getcwd() 

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
import numpy
import os

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")

# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=150, batch_size=10, verbose=0)

# evaluate the model
scores = model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[29]:


# 모형의 구조와 추정결과를 하나의 파일로 저장
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model

save_model(model, "Model/model.keras")


# In[31]:


# 모형의 로드
model = load_model('Model/model.keras')

# 로드된 모형의 요약
model.summary()


# In[32]:


# 데이터 불러오기
dataset = loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# 모형 평가
score = model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


# In[ ]:





# # 4. 최종모형 선택을 위한 체크포인트

# In[ ]:


# 모형의 검증 정확도가 개선될 때 추정치 체크
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.random.set_seed(12345)

# 자료 읽어오기
dataset = np.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# 모형 생성
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#모형 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# ## 1) 모형개선이 이루어질 때 체크포인트 저장

# In[ ]:


# 모형 정확도 개선때마다 체크 포인트(checkpoint)하여 파일로 저장
filepath="Model/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')

# 모형 추정
model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10,
          callbacks=[checkpoint], verbose=0)


# ## 2) 최고 성능 모형만 저장하기

# In[5]:


# 모형의 훈련과정에서 최고 성능 추정치만 저장
filepath="Model/weights.best.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# 모형 추정
model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10,
callbacks=callbacks_list, verbose=0)


# ## 3) EarlyStopping과 함께 사용하기

# In[ ]:


# 모형의 훈련과정에서 정확도가 악화될 때 훈련 중단
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=5)
callbacks_list = [checkpoint, early_stop]

# 모형 추정
model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10,
         callbacks=callbacks_list, verbose=0)


# # 다. 저장된 모형 불러오기

# In[ ]:


# 모형의 checkpoint 과정에서 저장된 추정치 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# 모형 생성
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모형 추정치(가중치) 불러오기
model.load_weights("Model/weights.best.keras")

# 모형 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Created model and loaded weights from file")

# 데이터 불러오기
dataset = np.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")

# 데이터를 입력 변수 input (X)와 출력변수 output (y)로 나누기
X = dataset[:,0:8]
y = dataset[:,8]

#불러온 가중치를 이용한 모형 평가
scores = model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:





# # 5. 모형 훈련과정의 시각적 추적 방법

# In[6]:


# 모형 훈련과정의 시각적 추적
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
dataset = np.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")

# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# 모형 생성
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모형 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모형 추정
history = model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)


# In[7]:


# 모형 훈련과정의 이력 키워드 출력
print(history.history.keys())


# In[8]:


# 정확도(accuracy) 이력 그래프 그리기
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[9]:


# 손실(loss) 이력 그래프 그리기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# # 6. 신경망 모형에서 활성화 함수

# In[17]:


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# x 값의 범위 설정
x = np.linspace(-5, 5, 100)

# 시그모이드 함수와 그래디언트 계산
y_sigmoid = sigmoid(x)
y_gradient = sigmoid_gradient(x)

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(x, y_sigmoid, label='Sigmoid Function', color='blue')
plt.plot(x, y_gradient, label='Sigmoid Gradient', color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.legend()
plt.title('Sigmoid Function and Its Gradient')
plt.xlabel('x')
plt.ylabel('Value')
plt.grid()
plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_gradient(x):
    return 1 - np.tanh(x) ** 2

# x 값의 범위 설정
x = np.linspace(-5, 5, 100)

# 하이퍼볼릭 탄젠트 함수와 그래디언트 계산
y_tanh = tanh(x)
y_gradient = tanh_gradient(x)

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(x, y_tanh, label='Tanh Function', color='blue')
plt.plot(x, y_gradient, label='Tanh Gradient', color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.legend()
plt.title('Hyperbolic Tangent Function and Its Gradient')
plt.xlabel('x')
plt.ylabel('Value')
plt.grid()
plt.show()


# In[19]:


import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def relu_gradient(x):
    return np.where(x > 0, 1, 0)

# x 값의 범위 설정
x = np.linspace(-5, 5, 100)

# ReLU 함수와 그래디언트 계산
y_relu = relu(x)
y_gradient = relu_gradient(x)

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(x, y_relu, label='ReLU Function', color='blue')
plt.plot(x, y_gradient, label='ReLU Gradient', color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.legend()
plt.title('ReLU Function and Its Gradient')
plt.xlabel('x')
plt.ylabel('Value')
plt.grid()
plt.show()


# In[20]:


import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_gradient(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# x 값의 범위 설정
x = np.linspace(-5, 5, 100)

# Leaky ReLU 함수와 그래디언트 계산
y_leaky_relu = leaky_relu(x)
y_gradient = leaky_relu_gradient(x)

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(x, y_leaky_relu, label='Leaky ReLU Function', color='blue')
plt.plot(x, y_gradient, label='Leaky ReLU Gradient', color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.legend()
plt.title('Leaky ReLU Function and Its Gradient')
plt.xlabel('x')
plt.ylabel('Value')
plt.grid()
plt.show()


# In[ ]:





# In[21]:


import numpy as np
import matplotlib.pyplot as plt

def mean_absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred)

def mean_absolute_error_gradient(y_true, y_pred):
    return np.where(y_pred > y_true, 1, -1)

# x 값의 범위 설정
y_true = 0  # 기준값 설정
x = np.linspace(-5, 5, 100)

# Mean Absolute Error 함수와 그래디언트 계산
y_mae = mean_absolute_error(y_true, x)
y_gradient = mean_absolute_error_gradient(y_true, x)

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(x, y_mae, label='Mean Absolute Error', color='blue')
plt.plot(x, y_gradient, label='MAE Gradient', color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.legend()
plt.title('Mean Absolute Error and Its Gradient')
plt.xlabel('Prediction (y_pred)')
plt.ylabel('Value')
plt.grid()
plt.show()


# In[ ]:





# In[22]:


import numpy as np
import matplotlib.pyplot as plt

def mean_square_error(y_true, y_pred):
    return (y_true - y_pred) ** 2

def mean_square_error_gradient(y_true, y_pred):
    return -2 * (y_true - y_pred)

# x 값의 범위 설정
y_true = 0  # 기준값 설정
x = np.linspace(-5, 5, 100)

# Mean Square Error 함수와 그래디언트 계산
y_mse = mean_square_error(y_true, x)
y_gradient = mean_square_error_gradient(y_true, x)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(x, y_mse, label='Mean Square Error', color='blue')
plt.plot(x, y_gradient, label='MSE Gradient', color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.legend()
plt.title('Mean Square Error and Its Gradient')
plt.xlabel('Prediction (y_pred)')
plt.ylabel('Value')
plt.grid()
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def log_likelihood_loss(y_true, y_pred):
    return -np.log(y_pred + 1e-9) * y_true

def log_likelihood_gradient(y_true, y_pred):
    return -y_true / (y_pred + 1e-9)

# x 값의 범위 설정
x = np.linspace(0.01, 0.99, 100).reshape(-1, 1)  # 0과 1 사이 값으로 설정

# 예제 데이터 (단일 클래스)
y_true = np.array([[1]])  # 정답 클래스

# Log-Likelihood Loss 함수와 그래디언트 계산
y_ll = log_likelihood_loss(y_true, x)
y_gradient = log_likelihood_gradient(y_true, x)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(x, y_ll, label='Log-Likelihood Loss', color='blue')
plt.plot(x, y_gradient, label='Log-Likelihood Gradient', color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.legend()
plt.title('Log-Likelihood Loss and Its Gradient')
plt.xlabel('Predicted Probability')
plt.ylabel('Value')
plt.grid()
plt.show()


# In[24]:


import numpy as np
import matplotlib.pyplot as plt

def categorical_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9), axis=1)

def categorical_cross_entropy_gradient(y_true, y_pred):
    return -y_true / (y_pred + 1e-9)

# x 값의 범위 설정
x = np.linspace(0.01, 0.99, 100).reshape(-1, 1)  # 0과 1 사이 값으로 설정

# 예제 데이터 (단일 클래스)
y_true = np.array([[1]])  # 정답 클래스

# Categorical Cross-Entropy 함수와 그래디언트 계산
y_cce = categorical_cross_entropy(y_true, x)
y_gradient = categorical_cross_entropy_gradient(y_true, x)

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(x, y_cce, label='Categorical Cross-Entropy', color='blue')
plt.plot(x, y_gradient, label='CCE Gradient', color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.legend()
plt.title('Categorical Cross-Entropy and Its Gradient')
plt.xlabel('Predicted Probability')
plt.ylabel('Value')
plt.grid()
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt

def sparse_categorical_cross_entropy(y_true, y_pred):
    return -np.log(y_pred[np.arange(len(y_true)), y_true] + 1e-9)

def sparse_categorical_cross_entropy_gradient(y_true, y_pred):
    grad = np.zeros_like(y_pred)
    grad[np.arange(len(y_true)), y_true] = -1 / (y_pred[np.arange(len(y_true)), y_true] + 1e-9)
    return grad

# x 값의 범위 설정
x = np.linspace(0.01, 0.99, 100).reshape(-1, 1)  # 0과 1 사이 값으로 설정

# 예제 데이터 (단일 클래스)
y_true = np.array([0])  # 정답 클래스 (0번째 인덱스)
y_pred = np.hstack([x, 1 - x])  # 두 클래스 확률 예측

# Sparse Categorical Cross-Entropy 함수와 그래디언트 계산
y_scce = np.tile(sparse_categorical_cross_entropy(y_true, y_pred), (100,))
y_gradient = sparse_categorical_cross_entropy_gradient(y_true, y_pred)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(x, y_scce, label='Sparse Categorical Cross-Entropy', color='blue')
plt.plot(x, y_gradient[:, 0], label='SCCE Gradient (Class 0)', color='red', linestyle='--')
plt.plot(x, y_gradient[:, 1], label='SCCE Gradient (Class 1)', color='green', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.legend()
plt.title('Sparse Categorical Cross-Entropy and Its Gradient')
plt.xlabel('Predicted Probability for Class 0')
plt.ylabel('Value')
plt.grid()
plt.show()



# ## Dropout on the Visible Layer

# In[9]:


# 입력층에서의 드롭아웃 적용
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 데이터 불러오기
dataframe = read_csv("Data/sonar3.csv", header=None)
dataset = dataframe.values

# 입력자료 input (X)와 출력자료 output (y) 정의
X = dataset[:,0:60].astype(float)
y = dataset[:,60]

# 출력자료 숫치화
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# 입력층에서 dropout 적용
def create_model():
    # 모형 생성
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dense(30, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dense(1, activation='sigmoid'))
    # 모형 컴파일
    sgd = SGD(learning_rate=0.1, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(model=create_model,
epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_y, cv=kfold)
print("Visible: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:





# ## Dropout on Hidden Layers

# In[10]:


# dropout in hidden layers with weight constraint
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(60, input_shape=(60,),
    activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    sgd = SGD(learning_rate=0.1, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(model=create_model,
                    epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## 9. 학습률 조정과 성과개선

# In[12]:


# 시간기반 학습률 스케줄(Time-Based Learning Rate Schedule)
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

# 자료 불러오기
dataframe = read_csv("Data/ionosphere.csv", header=None)
dataset = dataframe.values

#  입력자료 input (X)와 출력자료 output (y) 정의 
X = dataset[:,0:34].astype(float)
y = dataset[:,34]

# 출력자료 숫치화
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# 모형 생성
model = Sequential()
model.add(Dense(34, input_shape=(34,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모형 컴파일
epochs = 50
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate,
nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 모형 추정
model.fit(X, y, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2)


# In[ ]:





# In[14]:


# 감소기반 학습률 스케줄(Drop-Based Learning Rate Schedule)
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import LearningRateScheduler

# 학습률 
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# 데이터 읽어오기
dataframe = read_csv("Data/ionosphere.csv", header=None)
dataset = dataframe.values

#  입력자료 input (X)와 출력자료 output (y) 정의 
X = dataset[:,0:34].astype(float)
y = dataset[:,34]

# 출력자료 숫치화
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# 모형 생성
model = Sequential()
model.add(Dense(34, input_shape=(34,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모형 컴파일
sgd = SGD(learning_rate=0.0, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 콜백
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# 모형추정
model.fit(X, y, validation_split=0.33, epochs=50, batch_size=28,
callbacks=callbacks_list, verbose=2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




