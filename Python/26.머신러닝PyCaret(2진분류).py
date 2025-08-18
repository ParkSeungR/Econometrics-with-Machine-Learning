#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# In[36]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[37]:


# 라이브러리 불러오기
exec(open('Functions/Machine Learning_Econometrics_Lib.py').read())


# In[38]:


# 데이터 불러오기
filename = "Data/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)


# ## 1) 셋업(Setup)

# In[39]:


# 모형의 셋업
from pycaret.classification import *
s = setup(data, target = 'class', session_id = 12345)


# In[5]:


# 파라미터, 변수 리스트
get_config()


# In[6]:


# X_train_transformed 출력
get_config('X_train_transformed')


# In[7]:


# setup() 사용법
help(setup)


# In[33]:


# setup()에서 변수의 정규화(minmax)
s = setup(data, target = 'class', session_id = 12345,
          normalize = True, normalize_method = 'minmax')


# In[9]:


# X_train_transformed에서 특정변수 히스토그램
get_config('X_train_transformed')['preg'].hist()


# Notice that all the values are between 0 and 1 - that is because we passed `normalize=True` in the `setup` function. If you don't remember how it compares to actual data, no problem - we can also access non-transformed values using `get_config` and then compare. See below and notice the range of values on x-axis and compare it with histogram above.

# In[10]:


get_config('X_train')['preg'].hist()


# ___

# ## 2) 모형비교(Compare Models)

# In[11]:


# 2진분류 활용가능 모형
models()


# In[40]:


# 모형비교
best = compare_models()


# In[13]:


print(best)


# ## 3) 모형 생성과 평가(Create and Evaluate Model)

# In[14]:


# 로지스틱 회귀모형에 10겹 CV실행
lr = create_model('lr')


# The function above has return trained model object as an output. The scoring grid is only displayed and not returned. If you need access to the scoring grid you can use `pull` function to access the dataframe.

# In[15]:


lr_results = pull()
lr_results


# In[70]:


# create_model 활용법
help(create_model)


# In[41]:


# 모형의 평가
evaluate_model(best)

# 주의: minmax변환자료로 부터는 AUC가 구해지지 않음


# In[42]:


plot_model(best, plot = 'auc')


# In[43]:


plot_model(best, plot = 'confusion_matrix')


# ## 4) 모형 튜닝(Tune Model)

# In[18]:


# tune_model활용법
help(tune_model)


# In[19]:


# dt 모형의 생성
dt = create_model('dt')


# In[20]:


# 모형의 hyperparameters 튜닝
tuned_dt = tune_model(dt)


# In[21]:


# 설정된 tuning grid
dt_grid = {'max_depth' : [None, 2, 4, 6, 8, 10, 12]}

# 설정된 grid에 대한 튜닝(MAE 기준 평가)
tuned_dt = tune_model(dt, custom_grid = dt_grid, optimize = 'F1')


# ##  6) 앙상블 모형(Ensemble Model)

# In[22]:


# ensemble_model 활용법
help(ensemble_model)


# In[23]:


# 배깅(Bagging)
ensemble_model(dt, method = 'Bagging')


# In[24]:


# 부스팅(boosting)
ensemble_model(dt, method = 'Boosting')


# ## 7) 블랜딩 및 스태킹(Blend and Stacking Models)

# In[25]:


# recall기준 상위 3개 모형 선정
best_recall_models_top3 = compare_models(sort = 'recall', n_select = 3)


# In[89]:


best_recall_models_top3


# In[26]:


# 3개 상위모형의 블랜딩(blending top 3 models)
blend_models(best_recall_models_top3)


# In[91]:


# blend_models 사용법
help(blend_models)


# In[27]:


# 스태깅 모형(stacking models)
stack_models(best_recall_models_top3)


# In[93]:


help(stack_models)


# ## 8) 예측(Prediction)

# In[28]:


# 테스트 데이터에 대한 예측
predict_model(best)


# In[95]:


# 임의의 데이터에 대한 예측
predictions = predict_model(best, data=data)
predictions


# ## 9) 모형 저장과 로딩(Save and Load Model)

# In[96]:


# save model
save_model(best, './Output/Model_pima')


# In[97]:


# load model
loaded_from_disk = load_model('./Output/Model_pima')
loaded_from_disk

