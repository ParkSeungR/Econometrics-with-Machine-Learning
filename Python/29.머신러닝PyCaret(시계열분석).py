#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# Last updated: 16 Feb 2023
# 
# # 👋 PyCaret Time Series Forecasting Tutorial
# 
# PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that exponentially speeds up the experiment cycle and makes you more productive.
# 
# Compared with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with a few lines only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks, such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and a few more.
# 
# The design and simplicity of PyCaret are inspired by the emerging role of citizen data scientists, a term first used by Gartner. Citizen Data Scientists are power users who can perform both simple and moderately sophisticated analytical tasks that would previously have required more technical expertise.
# 

# # 💻 Installation
# 
# PyCaret is tested and supported on the following 64-bit systems:
# - Python 3.7 – 3.10
# - Python 3.9 for Ubuntu only
# - Ubuntu 16.04 or later
# - Windows 7 or later
# 
# You can install PyCaret with Python's pip package manager:
# 
# `pip install pycaret`
# 
# PyCaret's default installation will not install all the extra dependencies automatically. For that you will have to install the full version:
# 
# `pip install pycaret[full]`
# 
# or depending on your use-case you may install one of the following variant:
# 
# - `pip install pycaret[analysis]`
# - `pip install pycaret[models]`
# - `pip install pycaret[tuner]`
# - `pip install pycaret[mlops]`
# - `pip install pycaret[parallel]`
# - `pip install pycaret[test]`

# In[4]:


get_ipython().system('conda create -n pycaret_env python=3.11')


# In[ ]:


get_ipython().system('conda activate pycaret_env')
get_ipython().system('pip install --upgrade pip setuptools wheel')
get_ipython().system('pip install pycaret==3.3.2')
get_ipython().system('pip install ipykernel')
get_ipython().system('python -m ipykernel install --user --name=pycaret_env --display-name "Python (pycaret_env)"')
# deactivate


# In[1]:


get_ipython().system('pip install pycaret==3.3.2')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


# check installed version
import pycaret
pycaret.__version__


# # 🚀 Quick start

# PyCaret's time series forecasting module is now available. The module currently is suitable for univariate / multivariate time series forecasting tasks. The API of time series module is consistent with other modules of PyCaret.
# 
# It comes built-in with preprocessing capabilities and over 30 algorithms comprising of statistical / time-series methods as well as machine learning based models. In addition to the model training, this module has lot of other capabilities such as automated hyperparameter tuning, ensembling, model analysis, model packaging and deployment capabilities.
# 
# A typical workflow in PyCaret consist of following 5 steps in this order:
# 
# ### **Setup** ➡️ **Compare Models** ➡️ **Analyze Model** ➡️ **Prediction** ➡️ **Save Model** <br/>

# In[3]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWDirectory\MyStock")
# 현재 작업공간(working directory)확인  
os.getcwd() 


# In[9]:


import sys
print(sys.version)
get_ipython().system('python --version')


# In[13]:


import time
import numpy as np
import pandas as pd

from pycaret.time_series import TSForecastingExperiment


# In[3]:


# 한국의 거시경제 통계자료 불러오기
data = pd.read_csv('Data/Korea_GDP.csv',index_col='Time', parse_dates=True)

new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)

df = data.loc[data.index >= '2000-01-31', 'gdp']
df


# In[4]:


# plot the dataset
df.plot()


# ## Setup
# This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function in PyCaret. `Setup` has only one required parameter i.e. `data`. All the other parameters are optional.

# In[5]:


# 셋업(setup)
from pycaret.time_series import *
s = setup(df, fh = 4, session_id = 12345)


# Once the setup has been successfully executed it shows the information grid containing experiment level information.
# 
# - **Session id:**  A pseudo-random number distributed as a seed in all functions for later reproducibility. If no `session_id` is passed, a random number is automatically generated that is distributed to all functions.<br/>
# <br/>
# - **Approach:**  Univariate or multivariate. <br/>
# <br/>
# - **Exogenous Variables:**  Exogeneous variables to be used in model. <br/>
# <br/>
# - **Original data shape:**  Shape of the original data prior to any transformations. <br/>
# <br/>
# - **Transformed train set shape :**  Shape of transformed train set <br/>
# <br/>
# - **Transformed test set shape :**  Shape of transformed test set <br/>
# <br/>

# PyCaret has two set of API's that you can work with. (1) Functional (as seen above) and (2) Object Oriented API.
# 
# With Object Oriented API instead of executing functions directly you will import a class and execute methods of class.

# ## Check Stats
# The `check_stats` function is used to get summary statistics and run statistical tests on the original data or model residuals.

# In[36]:


# 데이터세트에 요약통계 및 안정성 검정
check_stats()


# ## Compare Models
# 
# This function trains and evaluates the performance of all the estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

# In[6]:


# 다양한 앨고리즘을 이용한 모형 추정
best = compare_models()


# ## Analyze Model

# You can use the `plot_model` function to analyzes the performance of a trained model on the test set. It may require re-training the model in certain cases.

# In[38]:


# plot forecast
plot_model(best, plot = 'forecast')


# In[39]:


# plot forecast for 12 Quarters in future
plot_model(best, plot = 'forecast', data_kwargs = {'fh' : 12})


# In[40]:


# residuals plot
plot_model(best, plot = 'residuals')


# In[41]:


# check docstring to see available plots
help(plot_model)


# An alternate to `plot_model` function is `evaluate_model`. It can only be used in Notebook since it uses ipywidget.

# ## Prediction
# The `predict_model` function returns `y_pred`. When data is `None` (default), it uses `fh` as defined during the `setup` function.

# In[25]:


# predict on test set
holdout_pred = predict_model(best)


# In[26]:


# show predictions df
holdout_pred.head()


# In[27]:


# generate forecast for 36 period in future
predict_model(best, fh = 12)


# ## Save Model

# Finally, you can save the entire pipeline on disk for later use, using pycaret's `save_model` function.

# In[28]:


# save pipeline
save_model(best, './Output/Model_timeseries')


# In[30]:


# load pipeline
loaded_best_pipeline = load_model('./Output/Model_timeseries')
loaded_best_pipeline


# # 👇 Detailed function-by-function overview

# ## ✅ Setup
# This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function in PyCaret. `Setup` has only one required parameter i.e. `data`. All the other parameters are optional.

# In[7]:


s = setup(df, fh = 4, session_id = 12345)


# To access all the variables created by the setup function such as transformed dataset, random_state, etc. you can use `get_config` method.

# In[44]:


# 데이터, 파라미터 접근
get_config()


# In[45]:


# y_train_transformed 출력
get_config('y_train_transformed')


# All the preprocessing configurations and experiment settings/parameters are passed into the `setup` function. To see all available parameters, check the docstring:

# In[8]:


help(setup)


# In[49]:


# init setup fold_strategy = expanding
s = setup(df, fh = 4, session_id = 12345,
          fold_strategy = 'expanding', numeric_imputation_target = 'drift')


# ## ✅ Compare Models
# This function trains and evaluates the performance of all estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

# In[50]:


best = compare_models()


# `compare_models` by default uses all the estimators in model library (all except models with `Turbo=False`) . To see all available models you can use the function `models()`

# In[51]:


# 이용가능 모형
models()


# You can use the `include` and `exclude` parameter in the `compare_models` to train only select model or exclude specific models from training by passing the model id's in `exclude` parameter.

# In[52]:


compare_ts_models = compare_models(include = ['ets', 'arima', 'theta', 'naive', 'snaive', 'grand_means', 'polytrend'])


# In[53]:


compare_ts_models


# The function above has return trained model object as an output. The scoring grid is only displayed and not returned. If you need access to the scoring grid you can use `pull` function to access the dataframe.

# In[54]:


compare_ts_models_results = pull()
compare_ts_models_results


# By default `compare_models` return the single best performing model based on the metric defined in the `sort` parameter. Let's change our code to return 3 top models based on `MAE`.

# In[55]:


best_r2_models_top3 = compare_models(sort = 'R2', n_select = 3)


# In[56]:


# list of top 3 models by MAE
best_r2_models_top3


# Some other parameters that you might find very useful in `compare_models` are:
# 
# - fold
# - cross_validation
# - budget_time
# - errors
# - parallel
# - engine
# 
# You can check the docstring of the function for more info.

# In[57]:


help(compare_models)


# ## ✅  Check Stats
# The `check_stats` function is used to get summary statistics and run statistical tests on the original data or model residuals.

# In[58]:


# check stats on original data
check_stats()


# In[59]:


# check_stats on residuals of best model
check_stats(estimator = best)


# ## ✅ Create Model
# This function trains and evaluates the performance of a given estimator using cross-validation. The output of this function is a scoring grid with CV scores by fold. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function. All the available models can be accessed using the models function.

# In[62]:


# check all the available models
models()


# In[63]:


# train ets with default fold=3
ets = create_model('ets')


# The function above has return trained model object as an output. The scoring grid is only displayed and not returned. If you need access to the scoring grid you can use `pull` function to access the dataframe.

# In[64]:


ets_results = pull()
print(type(ets_results))
ets_results


# In[65]:


# train theta model with fold=5
theta = create_model('theta', fold=5)


# In[66]:


# train theta with specific model parameters
create_model('theta', deseasonalize = False, fold=5)


# Some other parameters that you might find very useful in `create_model` are:
# 
# - cross_validation
# - engine
# - fit_kwargs
# 
# You can check the docstring of the function for more info.

# In[67]:


help(create_model)


# ## ✅ Tune Model
# 
# The `tune_model` function tunes the hyperparameters of the model. The output of this function is a scoring grid with cross-validated scores by fold. The best model is selected based on the metric defined in optimize parameter. Metrics evaluated during cross-validation can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

# In[68]:


# train a dt model with default params
dt = create_model('dt_cds_dt')


# In[69]:


# tune hyperparameters of dt
tuned_dt = tune_model(dt)


# Metric to optimize can be defined in `optimize` parameter (default = 'MASE'). Also, a custom tuned grid can be passed with `custom_grid` parameter.

# In[70]:


dt


# In[71]:


# define tuning grid
dt_grid = {'regressor__max_depth' : [None, 2, 4, 6, 8, 10, 12]}

# tune model with custom grid and metric = MAE
tuned_dt = tune_model(dt, custom_grid = dt_grid, optimize = 'MAE')


# In[72]:


# see tuned_dt params
tuned_dt


# In[ ]:


# to access the tuner object you can set return_tuner = True
tuned_dt, tuner = tune_model(dt, return_tuner=True)


# In[ ]:


# model object
tuned_dt


# In[ ]:


# tuner object
tuner


# For more details on all available `search_library` and `search_algorithm` please check the docstring. Some other parameters that you might find very useful in `tune_model` are:
# 
# - choose_better
# - custom_scorer
# - n_iter
# - search_algorithm
# - optimize
# 
# You can check the docstring of the function for more info.

# In[ ]:


# help(tune_model)


# ## ✅ Blend Models

# This function trains a `EnsembleForecaster` for select models passed in the estimator_list parameter. The output of this function is a scoring grid with CV scores by fold. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

# In[73]:


# top 3 models based on mae
best_r2_models_top3


# In[74]:


# blend top 3 models
blend_models(best_r2_models_top3)


# Some other parameters that you might find very useful in `blend_models` are:
# 
# - choose_better
# - method
# - weights
# - fit_kwargs
# - optimize
# 
# You can check the docstring of the function for more info.

# In[75]:


help(blend_models)


# ## ✅ Plot Model

# This function analyzes the performance of a trained model on the hold-out set. It may require re-training the model in certain cases.

# In[9]:


# plot forecast
plot_model()


# In[10]:


# plot acf
# for certain plots you don't need a trained model
plot_model(plot = 'acf')


# In[11]:


# plot acf
# for certain plots you don't need a trained model
plot_model(plot = 'pacf')


# In[12]:


# plot diagnostics
# for certain plots you don't need a trained model
plot_model(plot = 'diagnostics', fig_kwargs={"height": 600, "width": 900})


# In[13]:


# First, classical decomposition
# By default the seasonal period is the one detected during setup - 4 in this case.
plot_model(plot="decomp", fig_kwargs={"height": 500})


# In[14]:


# Users can change the seasonal period to explore what is best for this model.
plot_model(plot="decomp", data_kwargs={'seasonal_period': 4}, fig_kwargs={"height": 500})


# In[15]:


# Users may wish to customize the decomposition, for example, in this case multiplicative seasonality
# probably makes more sense since the magnitide of the seasonality increase as the time progresses
plot_model(plot="decomp", data_kwargs={'type': 'multiplicative'}, fig_kwargs={"height": 500})


# In[16]:


# Users can also plot STL decomposition
# Reference: https://otexts.com/fpp2/stl.html
plot_model(plot="decomp_stl", fig_kwargs={"height": 500})


# Some other parameters that you might find very useful in `plot_model` are:
# 
# - fig_kwargs
# - data_kwargs
# - display_format
# - return_fig
# - return_data
# - save
# 
# You can check the docstring of the function for more info.

# In[17]:


help(plot_model)


# ## ✅ Finalize Model
# This function trains a given model on the entire dataset including the hold-out set.

# In[80]:


final_best = finalize_model(best)


# In[81]:


final_best


# In[92]:


# generate forecast for 12 period in future
predict_model(final_best, fh = 24)


# In[93]:


# plot forecast for 12 Quarters in future
plot_model(final_best, plot = 'forecast', data_kwargs = {'fh' : 24})


# ## ✅ Save / Load Model
# This function saves the transformation pipeline and a trained model object into the current working directory as a pickle file for later use.

# In[ ]:


# save model
save_model(best, 'my_first_model')


# In[ ]:


# load model
loaded_from_disk = load_model('my_first_model')
loaded_from_disk


# # 외생변수를 가진 단변량 시계열 모형

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
# 현재 작업공간(working directory)확인  
os.getcwd() 


# In[2]:


import numpy as np
import pandas as pd

from pycaret.time_series import TSForecastingExperiment


# In[3]:


# 한국의 거시경제 통계자료 불러오기
data = pd.read_csv('./Data/Korea_GDP.csv',index_col='Time', parse_dates=True)

new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index

data.drop(columns=['exch', 'cpi', 'rat'], inplace=True)
print(data)


# In[4]:


# PyCaret을 사용한 자동화된 시계열 모델링
exp_auto = TSForecastingExperiment()
exp_auto.setup(data=data, target='con', seasonal_period=4)


# In[5]:


# 최적 모델 비교 및 선택
best_model = exp_auto.compare_models()


# In[6]:


# 자동화된 모델의 미래 예측
future_auto_predictions = exp_auto.predict_model(best_model, fh=4)


# In[7]:


# 예측 결과 출력
print(future_auto_predictions)

