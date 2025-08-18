#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:/JupyterWDirectory/MyStock")
# 현재 작업공간(working directory)확인  
os.getcwd() 

import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# # 피마 인디언 당뇨병 데이터세트(Pima Indians Diabetes Dataset)

# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터t세트 불러오기
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
               ]
df = pd.read_csv('Data/pima-indians-diabetes.data.csv', header=None, names=column_names)
df.info()
display(df)

X = df.drop("Outcome", axis=1)  
y = df["Outcome"] 

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=12345)

# 모형 정의
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))  # 첫 번째 은닉층
model.add(Dense(8, activation='relu'))                     # 두 번째 은닉층
model.add(Dense(1, activation='sigmoid'))                  # 출력층

# 모형 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모형 학습
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=10)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print('정확도: %.2f' % (accuracy * 100))

# 예측 수행
predictions = model.predict(X_test)
binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]
display(predictions)
display(binary_predictions)


# In[ ]:





# # 다중분류(multi-class classification with Keras): IRIS Dataset

# In[ ]:





# # Classification of Sonar Returns

# In[7]:


# Binary Classification with Sonar Dataset: Baseline
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
# load dataset
dataframe = read_csv("Data/sonar3.csv", header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
y = dataset[:,60]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)


# In[8]:


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_shape=(60,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[9]:


# evaluate model with standardized dataset
estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:


# 예측 수행
predictions = model.predict(X_test)
binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]


# In[ ]:





# # Regression Example With Boston Dataset: Standardized and Wider

# In[21]:


from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# 데이터 불러오기
dataframe = read_csv("Data/housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
X = dataset[:, 0:13]
y = dataset[:, 13]

# 출력 변수 스케일링
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# X, Y 각각 스케일링
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 모형 정의
def wider_model():
    model = Sequential()
    model.add(Dense(20, input_shape=(13,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Pipeline을 이용한 교차검증
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(f"Cross-validation mean MSE: {-results.mean():.4f}, std: {results.std():.4f}")

# 전체 데이터로 최종 모델 학습
final_model = wider_model()
final_model.fit(X_scaled, y_scaled, epochs=100, batch_size=5, verbose=0)

# 새로운 데이터 예측
new_data = np.array([[0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98]])
new_data_scaled = scaler_X.transform(new_data)
predicted_price_scaled = final_model.predict(new_data_scaled)

# 예측값을 원래 스케일로 복원
predicted_price = scaler_y.inverse_transform(predicted_price_scaled)

print("Predicted house price:", predicted_price[0][0])


# In[24]:


print("Predicted house price:", predicted_price[0][0])


# In[ ]:





# # MNIST 데이터세트을 이용한 숫자 이미지 인식

# In[ ]:


import tensorflow 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 이미지를 784차원 벡터로 펼치기
X_train = X_train.reshape((X_train.shape[0], 28 * 28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28 * 28)).astype('float32')

# 픽셀 값을 0-1 범위로 정규화
X_train /= 255
X_test /= 255

# 레이블을 원-핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
display(y_train, X_train)

# 모형 정의
model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모형 학습
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# 모형 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss * 100:.2f}%")

# 모형 예측
predictions = model.predict(X_test)
print(predictions)

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='mnist_model.png',
    show_shapes=True, show_dtype=True, show_layer_names=True,
    expand_nested=True, show_layer_activations=True)


# In[ ]:





# # 최적화 앨고리즘 튜닝

# ## 1. 배치 사이즈 및 에포크 수 튜닝

# In[5]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:/JupyterWDirectory/MyStock")
# 현재 작업공간(working directory)확인  
os.getcwd() 

import warnings
warnings.filterwarnings("ignore")


# In[6]:


# 배치 사이즈 및 에포크 수 튜닝을 위한 scikit-learn 활용
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

# KerasClassifier 활용을 위한 모형 생성
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 동일한 결과 재생을 위한 random seed 부여
tf.random.set_seed(12345)

# 데이터 불러오기
dataset = np.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")

# 투입변수 input (X)와 산출변수 output (y) 
X = dataset[:,0:8]
y = dataset[:,8]

# 모형 정의
model = KerasClassifier(model=create_model, verbose=0)

# 그리드 서치를 위한 배치사이즈와 에포크 수 정의
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# 분석결과의 출력
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    


# In[ ]:





# In[ ]:





# ## 2. 최적화 앨고리즘 튜닝

# In[8]:


# 최적화 앨고리즘 튜닝을 위한 scikit-learn 활용
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

# KerasClassifier 활용을 위한 모형 생성
def create_model(optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# 동일한 결과 재생을 위한 random seed 부여
tf.random.set_seed(12345)

# 데이터 불러오기
dataset = np.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")

# 투입변수 input (X)와 산출변수 output (y) 
X = dataset[:,0:8]
y = dataset[:,8]

# 모형 정의
model = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0)

# # 그리드 서치를 위한 옵티마이져 정의 
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(model__optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# 분석결과의 출력
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:





# ## 3. Learning Rate and Momentum

# In[9]:


# 학습률과 모멘텀 튜닝을 위한 scikit-learn 활용
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from scikeras.wrappers import KerasClassifier

# KerasClassifier 활용을 위한 모형 생성
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# 동일한 결과 재생을 위한 random seed 부여
tf.random.set_seed(12345)

# 데이터 불러오기
dataset = np.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")

# 투입변수 input (X)와 산출변수 output (y) 
X = dataset[:,0:8]
y = dataset[:,8]

# 모형 정의
model = KerasClassifier(model=create_model, loss="binary_crossentropy",
                        optimizer="SGD", epochs=100, batch_size=10, verbose=0)

# 그리드 서치를 위한 학습률 및 모멘텀 정의
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(optimizer__learning_rate=learn_rate, optimizer__momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# 분석결과의 출력
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:





# ## 4. Network Weight Initialization

# In[10]:


# 추정 파라미터의 초기치 튜닝을 위한 scikit-learn 활용
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

# KerasClassifier 활용을 위한 모형 생성
def create_model(init_mode='uniform'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), kernel_initializer=init_mode,
    activation='relu'))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 동일한 결과 재생을 위한 random seed 부여
tf.random.set_seed(12345)

# 데이터 불러오기
dataset = np.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")

# 투입변수 input (X)와 산출변수 output (y) 
X = dataset[:,0:8]
y = dataset[:,8]

# 모형 정의
model = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0)

# 그리드 서치를 위한 초기갑 부여방법의 정의
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
             'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(model__init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# 분석결과의 출력
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:





# In[ ]:





# ## 5. Neuron Activation Function

# In[11]:


# 활성화 함수 튜닝을 위한 scikit-learn 활용
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

# KerasClassifier 활용을 위한 모형 생성
def create_model(activation='relu'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), kernel_initializer='uniform',
    activation=activation))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 동일한 결과 재생을 위한 random seed 부여
tf.random.set_seed(12345)

# 데이터 불러오기
dataset = np.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")

# 투입변수 input (X)와 산출변수 output (y) 
X = dataset[:,0:8]
y = dataset[:,8]

# 모형 정의
model = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0)

# 그리드 서치를 위한 활성화 함수 정의
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid',
              'hard_sigmoid', 'linear']
param_grid = dict(model__activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)

# 분석결과의 출력
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:





# ## 6. Dropout Regularization

# In[12]:


# Dropout Regularization을 위한 scikit-learn 활용
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from scikeras.wrappers import KerasClassifier

# KerasClassifier 활용을 위한 모형 생성
def create_model(dropout_rate, weight_constraint):
    # create model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), kernel_initializer='uniform',
    activation='linear', kernel_constraint=MaxNorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 동일한 결과 재생을 위한 random seed 부여
tf.random.set_seed(12345)

# 데이터 불러오기
dataset = np.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")
print(dataset.dtype, dataset.shape)

# 투입변수 input (X)와 산출변수 output (y) 
X = dataset[:,0:8]
y = dataset[:,8]

# 모형 정의
model = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0)

# # 그리드 서치를 위한 제약조건과 dropout_rate 정의
weight_constraint = [1.0, 2.0, 3.0, 4.0, 5.0]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(model__dropout_rate=dropout_rate,
                  model__weight_constraint=weight_constraint)

#param_grid = dict(model__dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# 분석결과의 출력
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:





# ## 7. Number of Neurons in the Hidden Layer

# In[13]:


# 은닉층 수 튜닝을 위한 scikit-learn 활용
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.constraints import MaxNorm

# KerasClassifier 활용을 위한 모형 생성
def create_model(neurons):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_shape=(8,), kernel_initializer='uniform',
                    activation='linear', kernel_constraint=MaxNorm(4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 동일한 결과 재생을 위한 random seed 부여
tf.random.set_seed(12345)

# 데이터 불러오기
dataset = np.loadtxt("Data/pima-indians-diabetes.data.csv", delimiter=",")

# 투입변수 input (X)와 산출변수 output (y) 
X = dataset[:,0:8]
y = dataset[:,8]

# 모형 정의
model = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0)

# 그리드 서치를 위한 뉴런 수 정의
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(model__neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# 분석결과의 출력
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




