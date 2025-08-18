#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[2]:


# 라이브러리 불러오기
exec(open('Functions/Machine Learning_Econometrics_Lib.py').read())


# In[3]:


from pandas import read_csv
name = ('SepalLengthCm SepalWidthCm PetalLengthCm PetalWidthCm Species').split()
data = read_csv('./Data/iris.csv', usecols=name)
display(data)


# In[4]:


# 모형의 셋업
from pycaret.classification import *
model = setup(data, target = 'Species', session_id = 12345)


# In[5]:


# 파라미터, 변수 리스트
get_config()


# In[6]:


# X_train_transformed 출력
get_config('X_train_transformed')


# In[7]:


# setup() 사용법
help(setup)


# In[8]:


# setup()에서 변수의 정규화(minmax)
s = setup(data, target = 'Species', session_id = 12345,
          normalize = True, normalize_method = 'minmax')


# In[9]:


# X_train_transformed에서 특정변수 히스토그램
get_config('X_train_transformed')['SepalLengthCm'].hist()


# In[10]:


get_config('X_train')['SepalLengthCm'].hist()


# In[11]:


# 2) 모형비교(Compare Models)

# 2진분류 활용가능 모형
models()


# In[12]:


# 모형비교
best = compare_models()
print(best)


# `compare_models` by default uses all the estimators in model library (all except models with `Turbo=False`) . To see all available models you can use the function `models()`

# In[13]:


# 3) 모형 생성과 평가(Create and Evaluate Model)

# 로지스틱 회귀모형에 10겹 CV실행
lr = create_model('lr')
lr_results = pull()
lr_results


# In[48]:


# create_model 활용법
help(create_model)


# In[14]:


# 모형의 평가
evaluate_model(best)


# In[50]:


plot_model(best, plot = 'auc')


# In[51]:


plot_model(best, plot = 'confusion_matrix')


# The function above has return trained model object as an output. The scoring grid is only displayed and not returned. If you need access to the scoring grid you can use `pull` function to access the dataframe.

# In[52]:


# 4) 모형 튜닝(Tune Model)
# tune_model활용법
help(tune_model)


# In[53]:


# dt 모형의 생성
dt = create_model('dt')


# In[54]:


# 모형의 hyperparameters 튜닝
tuned_dt = tune_model(dt)


# In[55]:


# 설정된 tuning grid
dt_grid = {'max_depth' : [None, 2, 4, 6, 8, 10, 12]}
# 설정된 grid에 대한 튜닝(MAE 기준 평가)
tuned_dt = tune_model(dt, custom_grid = dt_grid, optimize = 'F1')


# In[56]:


#  6) 앙상블 모형(Ensemble Model)

# ensemble_model 활용법
help(ensemble_model)


# In[57]:


# 배깅(Bagging)
ensemble_model(dt, method = 'Bagging')

# 부스팅(boosting)
ensemble_model(dt, method = 'Boosting')


# In[58]:


# 7) 블랜딩 및 스태킹(Blend and Stacking Models)

# recall기준 상위 3개 모형 선정
best_recall_models_top3 = compare_models(sort = 'recall', n_select = 3)
best_recall_models_top3

# 3개 상위모형의 블랜딩(blending top 3 models)
blend_models(best_recall_models_top3)

# blend_models 사용법
help(blend_models)


# In[59]:


# 스태깅 모형(stacking models)
stack_models(best_recall_models_top3)
help(stack_models)


# In[60]:


# 8) 예측(Prediction)

# 테스트 데이터에 대한 예측
predict_model(best)

# 임의의 데이터에 대한 예측
predictions = predict_model(best, data=data)
predictions


# In[61]:


# 9) 모형 저장과 로딩(Save and Load Model)

# save model
save_model(best, './Output/Model_iris')


# In[30]:


# load model
loaded_from_disk = load_model('./Output/Model_iris')
loaded_from_disk

