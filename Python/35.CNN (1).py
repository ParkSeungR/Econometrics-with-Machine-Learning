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


# In[8]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.datasets.cifar10 import load_data

(X_train, y_train), (X_test, y_test) = load_data()

# 이미지 크기 조정 (정규화: 0~255 값을 0~1 범위로 변환)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# CNN(합성곱 신경망) 모형 정의
model = Sequential([
    Conv2D(32, (3,3), input_shape=(32, 32, 3), padding="same",
    activation="relu", kernel_constraint=MaxNorm(3)),
    Dropout(0.3),
    Conv2D(32, (3,3), padding="same",
    activation="relu", kernel_constraint=MaxNorm(3)),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation="relu", kernel_constraint=MaxNorm(3)),
    Dropout(0.5),
    Dense(10, activation="sigmoid")
                ])

# 모형 컴파일 (손실 함수 및 최적화 기법 설정)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy"])

# 모형 학습 수행
model.fit(X_train_scaled, y_train, epochs=1, batch_size=32,
        validation_data=(X_test_scaled, y_test))

# 모형 구조 출력
model.summary()


# In[9]:


# 입력 이미지 시각화
plt.imshow(X_train_scaled[7])
plt.show()


# In[10]:


# 각 층에서 출력된 특징 추출
extractor = tf.keras.Model(inputs=model.inputs,
                           outputs=[layer.output for layer in model.layers])
features = extractor(np.expand_dims(X_train_scaled[7], 0))

# 첫 번째 합성곱 층의 특징 맵 32개 시각화
l0_features = features[0].numpy()[0]
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(l0_features[..., i])
plt.show()

# 세 번째 층의 특징 맵 32개 시각화
l2_features = features[2].numpy()[0]
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(l2_features[..., i])
plt.show()


# In[21]:


# 세 번째 층의 특징 맵 64개 시각화
plt.figure(figsize=(8, 4))
for i in range(32):
    plt.subplot(4, 8, i+1)  # 8행 8열로 구성
    plt.imshow(l2_features[..., i])  # 회색조 시각화
    plt.axis('off')  # 축 제거로 깔끔하게 표시

plt.tight_layout()
plt.show()


# In[ ]:





# # 25.손글씨 숫자 인식 프로젝트

# In[2]:


# =============================
# 손글씨 숫자 인식 프로젝트
# =============================

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# MNIST 데이터셋 불러오기 (필요시 자동 다운로드)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 9개의 손글씨 숫자 이미지 출력 (회색조)
plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap='gray')
plt.tight_layout()
plt.show()


# In[7]:


# =============================
# 간단한 CNN 모형 구축 (MNIST)
# =============================

# 데이터셋 불러오기
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터 차원 변환 (샘플 개수, 너비, 높이, 채널)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# 픽셀 값을 0~1 범위로 정규화
X_train = X_train / 255
X_test = X_test / 255

# 출력값을 원-핫 인코딩으로 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# 간단한 CNN 모형 정의
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # 모형 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 모형 생성
model = baseline_model()

# 모형 학습 수행
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# 최종 평가
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN 오차율: %.2f%%" % (100-scores[1]*100))


# In[8]:


# =============================
# 더 큰 CNN 모형 구축 (MNIST)
# =============================

# 데이터셋 불러오기 및 전처리
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# 더 큰 CNN 모형 정의
def larger_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # 모형 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 모형 생성 및 학습 수행
model = larger_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200)

# 최종 평가
scores = model.evaluate(X_test, y_test, verbose=0)
print("더 큰 CNN 오차율: %.2f%%" % (100-scores[1]*100))


# In[9]:


# =============================
# 이미지 증강(Augmentation)을 통한 모형 성능 향상
# =============================

# 원본 이미지 출력 (기본 비교용)
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 데이터셋 불러오기
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 9개의 손글씨 숫자 이미지 출력 (회색조)
plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.axis('off') 
plt.tight_layout()
plt.show()


# In[10]:


# 특징 표준화 (Feature Standardization): 평균=0, 표준편차=1

from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 데이터 불러오기
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터 차원 변환 (샘플 개수, 너비, 높이, 채널)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# 정수 데이터를 실수형으로 변환
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 데이터 증강을 위한 설정
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

# 데이터셋의 평균과 표준편차를 이용하여 변환 적용
datagen.fit(X_train)

# 증강된 데이터 배치 하나만 가져와서 시각화
X_batch, y_batch = next(datagen.flow(X_train, y_train, batch_size=9, shuffle=False))

plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_batch[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[11]:


# 데이터 증강을 위한 설정 (ZCA Whitening 적용)
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             zca_whitening=True)

# 평균값을 기준으로 정규화 적용
X_mean = X_train.mean(axis=0)
datagen.fit(X_train - X_mean)

# 변환된 데이터셋 확인
X_centered = X_train - X_mean

# 증강된 데이터 배치 하나만 가져와서 시각화
X_batch, y_batch = next(datagen.flow(X_centered, y_train, batch_size=9, shuffle=False))

plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_batch[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[12]:


# 랜덤 회전 (Random Rotations): 최대 90도 회전
datagen = ImageDataGenerator(rotation_range=90)

# 증강된 데이터 배치 하나만 가져와서 시각화
X_batch, y_batch = next(datagen.flow(X_centered, y_train, batch_size=9, shuffle=False))

plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_batch[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[13]:


# 랜덤 이동 (Random Shifts): 너비 및 높이 방향으로 20% 이동
shift = 0.2
datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)

# 증강된 데이터 배치 하나만 가져와서 시각화
X_batch, y_batch = next(datagen.flow(X_centered, y_train, batch_size=9, shuffle=False))

plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_batch[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[14]:


# 랜덤 좌우 및 상하 뒤집기 (Random Flips)
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

# 증강된 데이터 배치 하나만 가져와서 시각화
X_batch, y_batch = next(datagen.flow(X_centered, y_train, batch_size=9, shuffle=False))

plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_batch[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.datasets.cifar10 import load_data

# 데이터 불러오기
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터 차원 변환 (샘플 개수, 너비, 높이, 채널)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# 정수 데이터를 실수형으로 변환
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 1. 데이터 증강 설정
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# 2. 데이터 준비
datagen.fit(X_train)

# 3. CNN 모델 구축
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 학습 (증강된 데이터 사용)
model.fit(datagen.flow(X_train.reshape(-1, 28, 28, 1), 
                      y_train, 
                      batch_size=32),
          epochs=1,
          validation_data=(X_test.reshape(-1, 28, 28, 1), y_test))


# In[5]:


# 6. 모형 구조 출력
model.summary()


# In[18]:


# 7. 최종 평가
scores = model.evaluate(X_test.reshape(-1, 28, 28, 1), y_test, verbose=0)
print("CNN 오차율: %.2f%%" % (100-scores[1]*100))


# In[ ]:





# # Project: Object Classification in Photographs

# In[ ]:


# 작업공간(working directory) 지정
import os  
os.chdir("E:/JupyterWDirectory/MyStock")

# 현재 작업공간(working directory) 확인
os.getcwd() 

import warnings
warnings.filterwarnings("ignore")


# In[19]:


# Simple CNN model for the CIFAR-10 Dataset
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one-hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=MaxNorm(3)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary(line_length=72)


# In[24]:


plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap='gray')
#    plt.axis('off')
plt.tight_layout()
plt.show()


# In[42]:


# 각 층에서 출력된 특징 추출
extractor = tf.keras.Model(inputs=model.inputs,
                           outputs=[layer.output for layer in model.layers])
features = extractor(np.expand_dims(X_train[7], 0))

# 첫 번째 합성곱 층의 특징 맵 32개 시각화
l0_features = features[0].numpy()[0]

plt.figure(figsize=(16, 8))
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(l0_features[..., i])
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[43]:


# 두번째 층의 특징 맵 32개 시각화
l2_features = features[2].numpy()[0]
plt.figure(figsize=(16, 8))
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(l2_features[..., i])
#    plt.axis('off')
plt.tight_layout()
plt.show()


# In[48]:


# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[53]:


# Korean Fonts
import matplotlib as mpl
mpl.rc('font', family='NanumGothic')
mpl.rc('axes', unicode_minus=False)
plt.figure(figsize=(8, 4))

# 정확도 시각화
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.title('모형 정확도')
plt.legend()
plt.show()


# In[56]:


# 손실 시각화
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.title('모형 손실')
plt.legend()
plt.show()


# In[58]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# 증강된 데이터로 학습
augmented_history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                              epochs=10, validation_data=(X_test, y_test))


# In[59]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# 예측 레이블 생성
pred_labels = np.argmax(model.predict(X_test), axis=1)
true_labels = np.argmax(y_test, axis=1)

# 혼동 행렬 생성
cm = confusion_matrix(true_labels, pred_labels)
ConfusionMatrixDisplay(cm, display_labels=["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"]).plot(cmap='Blues')
plt.title('혼동 행렬')
plt.show()


# In[ ]:





# In[ ]:





# In[8]:


# Large CNN model for the CIFAR-10 Dataset
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one-hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
epochs=epochs, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:





# In[1]:


import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding
import matplotlib.pyplot as plt
# load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)


# In[2]:


# summarize size
print("Training data: ")
print(X.shape)
print(y.shape)

# Summarize number of classes
print("Classes: ")
print(np.unique(y))


# In[3]:


# Summarize number of words
print("Number of words: ")
print(len(np.unique(np.hstack(X))))


# In[4]:


# Summarize review length
print("Review length: ")
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))

# plot review length
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result, bins=20)
plt.tight_layout()
plt.show()


# In[14]:


imdb.load_data(num_words=5000)
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)
Embedding(5000, 32, input_length=500)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


# MLP for the IMDB problem
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
epochs=1, batch_size=128, verbose=1)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[7]:


# CNN for the IMDB problem
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# pad dataset to a maximum review length in words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
epochs=1, batch_size=128, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




