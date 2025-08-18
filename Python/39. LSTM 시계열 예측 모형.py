#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 작업공간(working directory) 지정
import os  
os.chdir("E:/JupyterWDirectory/MyStock")

# 현재 작업공간(working directory) 확인
os.getcwd() 

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# 한국의 GDP증가율 예측을 위한 LSTM 모형 구축
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Bidirectional, TimeDistributed, ConvLSTM2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
from tensorflow.keras.callbacks import EarlyStopping

# 한글폰트
import matplotlib as mpl
mpl.rc('font', family='NanumGothic')
mpl.rc('axes', unicode_minus=False)

import sys
sys.path.append('Functions')
from TS_Generator import create_sequence_data, split_train_test, scale_data, inverse_transform_predictions


# In[3]:


# 랜덤시드 reproducibility
tf.random.set_seed(12345)

# 데이터세트 불러오기(1999Q1-2023Q4)
# 한국의 거시통계(소비, 투자, 정부지출, 수출, 수입, GDP, 소비자물가, M2, 이자율, 실업률
# (time	con	inv	gov	ex	im	gdp	cpi	m2	r	un)
dataframe = pd.read_csv('Data/Korea_macro.csv', usecols=range(1, 11))

dataframe['rgdp'] = dataframe['gdp'].pct_change(4) * 100
dataframe.dropna(inplace=True)
display(dataframe)


# In[4]:


# GDP 증가율
Gr_gdp = dataframe['rgdp'].values

# GDP 증가율 그래프
plt.figure(figsize=(12, 6))
plt.plot(Gr_gdp)
plt.title('GDP 증가율 추이')
plt.ylabel('GDP 증가율')
plt.xlabel('시간')
plt.grid(True)
plt.show()


# In[5]:


dataset = dataframe.values
dataset = dataset.astype('float32')
display(dataset[:5, :])


# In[6]:


# 입력 및 출력 시퀀스 길이 설정
input_seq_len = 4   # 시차 4
output_seq_len = 8  # 다음 4 예측

# 단변량 시계열 예측을 위한 데이터 구성(시차와 예측기간의 수 지정)
X, y = create_sequence_data(Gr_gdp, input_seq_len=input_seq_len, output_seq_len=output_seq_len)

# 변환된 데이터 출력
for i in range(5):
    print(X[i], y[i])


# In[7]:


# 훈련/테스트 데이터 분할 (시간 순서 유지)
X_train, X_test, y_train, y_test = split_train_test(X, y, test_ratio=0.2, shuffle=False)

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scale_data(X_train, X_test, y_train, y_test, scaler=scaler)

# y의 형태 확인
print(f"y_train_scaled shape: {y_train_scaled.shape}")
# X의 형태 확인
print(f"X_train_scaled shape: {X_train_scaled.shape}")

# 변환된 데이터 출력
for i in range(5):
    print(X_train_scaled[i], y_train_scaled[i])


# In[ ]:





# In[8]:


# Vanilla LSTM 모형 구축
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(input_seq_len, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(output_seq_len))  # 출력은 다음 8 시점

# 모형 컴파일
model.compile(optimizer='adam', loss='mse')
model.summary()


# In[9]:


# 조기 종료 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모형 추정
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
                    )


# In[10]:


# 훈련 과정 시각화
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid(True)
plt.show()


# In[11]:


# 예측
y_pred_scaled = model.predict(X_test_scaled)

# 예측 결과를 원래 스케일로 변환
y_pred = inverse_transform_predictions(y_pred_scaled, y_scaler)
y_test_original = inverse_transform_predictions(y_test_scaled, y_scaler)

# 테스트 세트에 대한 예측 시각화
plt.figure(figsize=(12, 6))

# 실제 값
actual_values = [y_test_original[i][0] for i in range(len(y_test_original))]
plt.plot(actual_values, label='Actual (First Value)', color='blue')

# 예측 값
predicted_values = [y_pred[i][0] for i in range(len(y_pred))]
plt.plot(predicted_values, label='Predicted (First Value)', color='red')

plt.title('GDP 증가율 예측')
plt.ylabel('GDP 증가율')
plt.xlabel('시간')
plt.legend()
plt.grid(True)
plt.show()


# In[12]:


# 평가 지표 계산
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 모든 테스트 샘플과 모든 예측 시점에 대한 오차 계산
mse = mean_squared_error(y_test_original.flatten(), y_pred.flatten())
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original.flatten(), y_pred.flatten())

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")


# In[13]:


# 향후 8 시점 예측
# 가장 최근 데이터에서 마지막 input_seq_len개 포인트를 가져와 예측
latest_data = Gr_gdp[-input_seq_len:].reshape(1, input_seq_len, 1)
latest_data_scaled = scaler.transform(latest_data.reshape(-1, 1)).reshape(1, input_seq_len, 1)

future_pred_scaled = model.predict(latest_data_scaled)
future_pred = y_scaler.inverse_transform(future_pred_scaled).reshape(-1)

# 향후 예측 시각화
plt.figure(figsize=(12, 6))

# 과거 데이터
plt.plot(range(len(Gr_gdp)), Gr_gdp, label='Historical GDP')

# 향후 예측
future_index = range(len(Gr_gdp), len(Gr_gdp) + output_seq_len)
plt.plot(future_index, future_pred, label='Future Prediction', color='red', marker='o')

plt.axvline(x=len(Gr_gdp)-1, color='green', linestyle='--', label='Forecast Start')
plt.title('향후 8기 GDP 증가율 예측')
plt.ylabel('GDP 증가율')
plt.xlabel('시간(분기)')
plt.legend()
plt.grid(True)
plt.show()


# In[14]:


print("향후 8 시점 GDP 증가율 예측치:")
for i, pred in enumerate(future_pred):
    print(f"시점 {i+1}: {pred:.4f}")


# In[ ]:





# # Bidirectional LSTM 모형 구축

# In[15]:


# Bidirectional LSTM 모형 구축
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(input_seq_len, 1)))
model.add(Dense(output_seq_len))  # 출력은 다음 4 시점

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')
model.summary()

# 조기 종료 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 훈련
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
                    )


# In[16]:


# 훈련 과정 시각화
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid(True)
plt.show()


# In[17]:


# 향후 8 시점 예측
# 가장 최근 데이터에서 마지막 input_seq_len개 포인트를 가져와 예측
latest_data = Gr_gdp[-input_seq_len:].reshape(1, input_seq_len, 1)
latest_data_scaled = scaler.transform(latest_data.reshape(-1, 1)).reshape(1, input_seq_len, 1)

future_pred_scaled = model.predict(latest_data_scaled)
future_pred = y_scaler.inverse_transform(future_pred_scaled).reshape(-1)

# 향후 예측 시각화
plt.figure(figsize=(12, 6))

# 과거 데이터
plt.plot(range(len(Gr_gdp)), Gr_gdp, label='Historical GDP')

# 향후 예측
future_index = range(len(Gr_gdp), len(Gr_gdp) + output_seq_len)
plt.plot(future_index, future_pred, label='Future Prediction', color='red', marker='o')

plt.axvline(x=len(Gr_gdp)-1, color='green', linestyle='--', label='Forecast Start')
plt.title('향후 8기 GDP 증가율 예측')
plt.ylabel('GDP 증가율')
plt.xlabel('시간(분기)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# # CNN-LSTM

# In[19]:


# 작업공간(working directory) 지정
import os  
os.chdir("E:/JupyterWDirectory/MyStock")

# 현재 작업공간(working directory) 확인
os.getcwd() 

import warnings
warnings.filterwarnings("ignore")

# 한국 GDP증가율 예측을 위한 CNN-LSTM 모형 구축
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Reshape, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping

import sys
sys.path.append('Functions')
from TS_Generator import create_sequence_data, split_train_test, scale_data, inverse_transform_predictions

# 랜덤시드 reproducibility
tf.random.set_seed(12345)

# 데이터세트 불러오기(1999Q1-2023Q4)
# 한국의 거시통계(소비, 투자, 정부지출, 수출, 수입, GDP, 소비자물가, M2, 이자율, 실업률
# (time	con	inv	gov	ex	im	gdp	cpi	m2	r	un)
dataframe = pd.read_csv('Data/Korea_macro.csv', usecols=range(1, 11))

dataframe['rgdp'] = dataframe['gdp'].pct_change(4) * 100
dataframe.dropna(inplace=True)
display(dataframe)

# GDP 증가율
Gr_gdp = dataframe['rgdp'].values

# GDP 증가율 그래프
plt.figure(figsize=(12, 6))
plt.plot(Gr_gdp)
plt.title('GDP 증가율 추이')
plt.ylabel('GDP 증가율')
plt.xlabel('시간')
plt.grid(True)
plt.show()

# 입력 및 출력 시퀀스 길이 설정
input_seq_len = 4  # 시차 8
output_seq_len = 8  # 다음 8 예측

# 시퀀스 데이터 생성
X, y = create_sequence_data(Gr_gdp, input_seq_len=input_seq_len, output_seq_len=output_seq_len)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# 훈련/테스트 데이터 분할 (시간 순서 유지)
X_train, X_test, y_train, y_test = split_train_test(X, y, test_ratio=0.2, shuffle=False)

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scale_data(X_train, X_test, y_train, y_test, scaler=scaler)


# In[65]:


# CNN-LSTM 모델 구축
model = Sequential()

# 1D CNN 레이어 추가 - padding='same'으로 변경하여 출력 크기 유지
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same', input_shape=(input_seq_len, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# 두 번째 CNN 레이어 - 작은 커널 사이즈와 padding='same' 사용
model.add(Conv1D(filters=32, kernel_size=1, activation='relu', padding='same'))

# MaxPooling 제거하여 더 이상의 시퀀스 길이 감소 방지
model.add(Dropout(0.2))

# LSTM 레이어 추가
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))

# 출력 레이어
model.add(Dense(output_seq_len))

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')
model.summary()


# In[66]:


# 조기 종료 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 훈련
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 훈련 과정 시각화
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid(True)
plt.show()


# In[69]:


# 예측
y_pred_scaled = model.predict(X_test_scaled)

# 예측 결과를 원래 스케일로 변환
y_pred = inverse_transform_predictions(y_pred_scaled, y_scaler)
y_test_original = inverse_transform_predictions(y_test_scaled, y_scaler)

# 테스트 세트에 대한 예측 시각화
plt.figure(figsize=(12, 6))

# 실제 값들 - 첫 번째 값들만 추출하여 플랏
actual_values = [y_test_original[i][0] for i in range(len(y_test_original))]
plt.plot(actual_values, label='Actual (First Value)', color='blue')

# 예측 값들 - 첫 번째 값들만 추출하여 플랏
predicted_values = [y_pred[i][0] for i in range(len(y_pred))]
plt.plot(predicted_values, label='Predicted (First Value)', color='red')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('GDP YoY Percent Change Forecast vs Actual')
plt.ylabel('Percent Change (%)')
plt.xlabel('Time Steps')
plt.legend()
plt.grid(True)
plt.show()

# 평가 지표 계산
# 모든 테스트 샘플과 모든 예측 시점에 대한 오차 계산
mse = mean_squared_error(y_test_original.flatten(), y_pred.flatten())
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original.flatten(), y_pred.flatten())

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")



# In[70]:


# 향후 8 시점 예측
# 가장 최근 데이터에서 마지막 input_seq_len개 포인트를 가져와 예측
latest_data = Gr_gdp[-input_seq_len:].reshape(1, input_seq_len, 1)
latest_data_scaled = scaler.transform(latest_data.reshape(-1, 1)).reshape(1, input_seq_len, 1)

future_pred_scaled = model.predict(latest_data_scaled)
future_pred = y_scaler.inverse_transform(future_pred_scaled).reshape(-1)

# 향후 예측 시각화
plt.figure(figsize=(12, 6))

# 과거 데이터
plt.plot(range(len(Gr_gdp)), Gr_gdp, label='Historical GDP')

# 향후 예측
future_index = range(len(Gr_gdp), len(Gr_gdp) + output_seq_len)
plt.plot(future_index, future_pred, label='Future Prediction', color='red', marker='o')

plt.axvline(x=len(Gr_gdp)-1, color='green', linestyle='--', label='Forecast Start')
plt.title('향후 8기 GDP 증가율 예측')
plt.ylabel('GDP 증가율')
plt.xlabel('시간(분기)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# # ConvLSTM

# In[71]:


from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization

# ConvLSTM을 위한 입력 형태 변환
# ConvLSTM은 5D 텐서를 입력으로 받음: [samples, time_steps, rows, cols, channels]
# 여기서는 rows=1, cols=1, channels=1로 설정
X_train_convlstm = X_train_scaled.reshape(X_train_scaled.shape[0], input_seq_len, 1, 1, 1)
X_test_convlstm = X_test_scaled.reshape(X_test_scaled.shape[0], input_seq_len, 1, 1, 1)

# ConvLSTM 모델 구축
model = Sequential()

# ConvLSTM 레이어 추가
model.add(ConvLSTM2D(filters=64, kernel_size=(1, 1), 
                     activation='relu', 
                     input_shape=(input_seq_len, 1, 1, 1),
                     return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=32, kernel_size=(1, 1), 
                     activation='relu',
                     return_sequences=False))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(output_seq_len))

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')
model.summary()


# In[72]:


# 조기 종료 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 훈련
history = model.fit(
    X_train_convlstm, y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 훈련 과정 시각화
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid(True)
plt.show()


# In[73]:


# 향후 8 시점 예측
# 가장 최근 데이터에서 마지막 input_seq_len개 포인트를 가져와 예측
latest_data = Gr_gdp[-input_seq_len:].reshape(1, input_seq_len, 1)
latest_data_scaled = scaler.transform(latest_data.reshape(-1, 1)).reshape(1, input_seq_len, 1)

future_pred_scaled = model.predict(latest_data_scaled)
future_pred = y_scaler.inverse_transform(future_pred_scaled).reshape(-1)

# 향후 예측 시각화
plt.figure(figsize=(12, 6))

# 과거 데이터
plt.plot(range(len(Gr_gdp)), Gr_gdp, label='Historical GDP')

# 향후 예측
future_index = range(len(Gr_gdp), len(Gr_gdp) + output_seq_len)
plt.plot(future_index, future_pred, label='Future Prediction', color='red', marker='o')

plt.axvline(x=len(Gr_gdp)-1, color='green', linestyle='--', label='Forecast Start')
plt.title('향후 8기 GDP 증가율 예측')
plt.ylabel('GDP 증가율')
plt.xlabel('시간(분기)')
plt.legend()
plt.grid(True)
plt.show()


# In[74]:


# 향후 8 시점 예측
# 가장 최근 데이터에서 마지막 input_seq_len개 포인트를 가져와 예측
latest_data = Gr_gdp[-input_seq_len:].reshape(1, input_seq_len, 1)
latest_data_scaled = scaler.transform(latest_data.reshape(-1, 1)).reshape(1, input_seq_len, 1)

# ConvLSTM은 5차원 입력이 필요: (samples, time_steps, rows, cols, features)
latest_data_scaled_conv = latest_data_scaled.reshape(1, input_seq_len, 1, 1, 1)
future_pred_scaled = model.predict(latest_data_scaled_conv)

# 출력도 형태 변환이 필요
# 모델 출력이 5차원이라면 3차원으로 변환
if len(future_pred_scaled.shape) == 5:
    future_pred_scaled = future_pred_scaled.reshape(1, output_seq_len, 1)

future_pred = y_scaler.inverse_transform(future_pred_scaled.reshape(-1, 1)).reshape(-1)

# 향후 예측 시각화
plt.figure(figsize=(12, 6))
# 과거 데이터
plt.plot(range(len(Gr_gdp)), Gr_gdp, label='Historical GDP')
# 향후 예측
future_index = range(len(Gr_gdp), len(Gr_gdp) + output_seq_len)
plt.plot(future_index, future_pred, label='Future Prediction', color='red', marker='o')
plt.axvline(x=len(Gr_gdp)-1, color='green', linestyle='--', label='Forecast Start')
plt.title('향후 8기 GDP 증가율 예측')
plt.ylabel('GDP 증가율')
plt.xlabel('시간(분기)')
plt.legend()
plt.grid(True)
plt.show()


# # 다변량 시계열 예측을 위한 데이터 구성(시차와 예측기간의 수 지정, 예측대상 변수 지정)

# In[48]:


# 작업공간(working directory) 지정
import os  
os.chdir("E:/JupyterWDirectory/MyStock")

# 현재 작업공간(working directory) 확인
os.getcwd() 

import warnings
warnings.filterwarnings("ignore")

# 다변량 예측을 위한 LSTM 모형 구축
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
from tensorflow.keras.callbacks import EarlyStopping

# 한글폰트
import matplotlib as mpl
mpl.rc('font', family='NanumGothic')
mpl.rc('axes', unicode_minus=False)

import sys
sys.path.append('Functions')
from TS_Generator import create_sequence_data, split_train_test, scale_data, inverse_transform_predictions

# 데이터세트 불러오기(1999Q1-2023Q4)
# 한국의 거시통계(소비, 투자, 정부지출, 수출, 수입, GDP, 소비자물가, M2, 이자율, 실업률
dataset = pd.read_csv('Data/Korea_macro.csv', usecols=range(1, 11))
dataset['Gr_gdp'] = dataset['gdp'].pct_change(4) * 100
dataset['Gr_con'] = dataset['con'].pct_change(4) * 100
dataset['Gr_inv'] = dataset['inv'].pct_change(4) * 100
dataset['Gr_gov'] = dataset['gov'].pct_change(4) * 100
dataset = dataset.filter(like='Gr_')
dataset.dropna(inplace=True)
print(dataset)

# 시각화 
plt.figure(figsize=(12, 6))
# 각 선에 레이블 추가
plt.plot(dataset['Gr_gdp'], label='GDP 증가율')
plt.plot(dataset['Gr_con'], label='소비 증가율')
plt.plot(dataset['Gr_inv'], label='투자 증가율')
plt.plot(dataset['Gr_gov'], label='정부지출 증가율')
plt.title('소비, 투자, 정부지출, 국민소득 증가율 추이')
plt.ylabel('주요 거시경제 변수')
plt.xlabel('시간(분기)')
plt.grid(True)
# 범례 추가
plt.legend(loc='best')  
plt.show()


# In[49]:


# 입력 및 출력 시퀀스 길이 설정
input_seq_len = 4  # 시차 4
output_seq_len = 8  # 다음 8 예측

# 시퀀스 데이터 생성
X, y = create_sequence_data(dataframe, input_seq_len=input_seq_len, output_seq_len=output_seq_len, target_columns=[0, 1, 2])
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 훈련/테스트 데이터 분할 (시간 순서 유지)
X_train, X_test, y_train, y_test = split_train_test(X, y, test_ratio=0.2, shuffle=False)

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scale_data(X_train, X_test, y_train, y_test, scaler=scaler)

# y의 형태 확인
print(f"y_train_scaled shape: {y_train_scaled.shape}")  

# 방법 1: Dense 출력층 수정 (타겟 컬럼 수에 맞게)
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(input_seq_len, X_train_scaled.shape[2])))
model.add(LSTM(100, activation='relu'))
model.add(Dense(output_seq_len * 3))  
model.add(tf.keras.layers.Reshape((output_seq_len, 3)))  

# 모형 컴파일
model.compile(optimizer='adam', loss='mse')
model.summary()

# 조기 종료 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 훈련
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 훈련 과정 시각화
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid(True)
plt.show()


# In[62]:


# 예측
y_pred_scaled = model.predict(X_test_scaled)

# 예측 결과를 원래 스케일로 변환
y_pred = inverse_transform_predictions(y_pred_scaled, y_scaler)
y_test_original = inverse_transform_predictions(y_test_scaled, y_scaler)

# 각 변수별로 시각화 (GDP, 소비, 투자)
variable_names = ['GDP', 'Consumption', 'Investment']

# 모든 테스트 샘플에 대한 예측 시각화 (각 변수별로)
plt.figure(figsize=(15, 12))

for var_idx in range(3):  # 3개 변수 (소비, 투자, GDP)
    plt.subplot(3, 1, var_idx+1)
    # 실제값
    actual_values = []
    for i in range(len(y_test_original)):
        actual_values.extend(y_test_original[i, :, var_idx])
    plt.plot(actual_values, label=f'Actual {variable_names[var_idx]}', color='blue')
    # 예측값
    predicted_values = []
    for i in range(len(y_pred)):
        predicted_values.extend(y_pred[i, :, var_idx])
    plt.plot(predicted_values, label=f'Predicted {variable_names[var_idx]}', color='red')
    
    plt.title(f'{variable_names[var_idx]} Forecast vs Actual')
    plt.ylabel(variable_names[var_idx])
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

# 평가 지표 계산 (각 변수별로)
from sklearn.metrics import mean_squared_error, mean_absolute_error

for var_idx in range(3):
    # 특정 변수에 대한 실제값과 예측값 추출
    y_true = y_test_original[:, :, var_idx].flatten()
    y_predicted = y_pred[:, :, var_idx].flatten()
    
    # 평가 지표 계산
    mse = mean_squared_error(y_true, y_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_predicted)
    
    print(f"\n{variable_names[var_idx]} 평가 지표:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")


# In[63]:


# 향후 예측을 위해 최신 데이터 준비
# 원본 데이터에서 최근 input_seq_len 개의 데이터 포인트 가져오기
latest_data = dataset[-input_seq_len:].values.reshape(1, input_seq_len, 4)

# 데이터 스케일링
latest_data_scaled = scaler.transform(latest_data.reshape(-1, 4)).reshape(1, input_seq_len, 4)

# 향후 output_seq_len 시점 예측
future_pred_scaled = model.predict(latest_data_scaled)
future_pred = inverse_transform_predictions(future_pred_scaled, y_scaler)

# 향후 예측 시각화 (각 변수별로)
plt.figure(figsize=(15, 12))

# 변수 열 이름 목록
column_names = dataset.columns[[0, 1, 2]]  # 첫 3개 열의 이름

for var_idx in range(3):
    plt.subplot(3, 1, var_idx+1)
    # 과거 데이터 (해당 변수만)
    column_name = column_names[var_idx]
    historical_data = dataset[column_name]
    plt.plot(range(len(historical_data)), historical_data, label=f'Historical {variable_names[var_idx]}')
    # 향후 예측
    future_index = range(len(historical_data), len(historical_data) + output_seq_len)
    plt.plot(future_index, future_pred[0, :, var_idx], label=f'Future {variable_names[var_idx]} Prediction', 
             color='red', marker='o')
    plt.axvline(x=len(historical_data)-1, color='green', linestyle='--', label='Forecast Start')
    plt.title(f'{variable_names[var_idx]} Forecast for Next 8 Time Steps')
    plt.ylabel(variable_names[var_idx])
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()


# In[61]:


# 향후 8 시점 예측값 출력 (각 변수별로)
for var_idx in range(3):
    print(f"\n향후 8 시점 {variable_names[var_idx]} 예측값:")
    for i in range(output_seq_len):
        print(f"시점 {i+1}: {future_pred[0, i, var_idx]:.4f}")

