#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# Last updated: 16 Feb 2023
# 
# # 👋 PyCaret Clustering Tutorial
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

# In[1]:


# check installed version
import pycaret
pycaret.__version__


# # 🚀 Quick start

# PyCaret's Clustering Module is an unsupervised machine learning module that performs the task of grouping a set of objects in such a way that objects in the same group (also known as a cluster) are more similar to each other than to those in other groups.
# 
# It provides several pre-processing features that prepare the data for modeling through the setup function. It has over 10 ready-to-use algorithms and several plots to analyze the performance of trained models.
# 
# A typical workflow in PyCaret's unsupervised module consist of following 6 steps in this order:
# 
# ### **Setup** ➡️ **Create Model** ➡️ **Assign Labels** ➡️ **Analyze Model** ➡️ **Prediction** ➡️ **Save Model**

# In[2]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[3]:


# 라이브러리 불러오기
exec(open('Functions/Machine Learning_Econometrics_Lib.py').read())


# In[4]:


# loading sample dataset from pycaret dataset module
from pycaret.datasets import get_data
data = get_data('jewellery')


# In[5]:


from pandas import read_csv
name = ('Age Income SpendingScore Savings').split()
data = read_csv('./Data/jewellery.csv', usecols=name)
display(data)


# ## Setup
# This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function in PyCaret. It only has one required parameter i.e. `data`. All the other parameters are optional.

# In[6]:


# import pycaret clustering and init setup
from pycaret.clustering import *
model = setup(data, normalize = True)


# Once the setup has been successfully executed it shows the information grid containing experiment level information.
# 
# - **Session id:**  A pseudo-random number distributed as a seed in all functions for later reproducibility. If no `session_id` is passed, a random number is automatically generated that is distributed to all functions.<br/>
# <br/>
# - **Original data shape:**  Shape of the original data prior to any transformations. <br/>
# <br/>
# - **Transformed data shape:**  Shape of data after transformations <br/>
# <br/>
# - **Numeric features :**  The number of features considered as numerical. <br/>
# <br/>
# - **Categorical features :**  The number of features considered as categorical. <br/>

# PyCaret has two set of API's that you can work with. (1) Functional (as seen above) and (2) Object Oriented API.
# 
# With Object Oriented API instead of executing functions directly you will import a class and execute methods of class.

# ## Create Model
# 
# This function trains and evaluates the performance of a given model. Metrics evaluated can be accessed using the `get_metrics` function. Custom metrics can be added or removed using the `add_metric` and `remove_metric` function. All the available models can be accessed using the `models` function.

# In[7]:


# to check all the available models
models()


# In[8]:


# train kmeans model
kmeans = create_model('kmeans')


# In[9]:


# train meanshift model
meanshift = create_model('meanshift')


# In[10]:


#모형분석(Analyze Model)
evaluate_model(kmeans)


# In[11]:


# plot pca cluster plot
plot_model(kmeans, plot = 'cluster')


# In[11]:


# functional API
plot_model(kmeans, plot = 'elbow')


# In[12]:


# functional API
plot_model(kmeans, plot = 'silhouette')


# ## Assign Model
# This function assigns cluster labels to the training data, given a trained model.

# In[13]:


kmeans_cluster = assign_model(kmeans)
kmeans_cluster


# ## Analyze Model

# You can use the `plot_model` function to analyzes the performance of a trained model on the test set. It may require re-training the model in certain cases.

# In[16]:


# check docstring to see available plots
help(plot_model)


# An alternate to `plot_model` function is `evaluate_model`. It can only be used in Notebook since it uses ipywidget.

# ## Prediction
# The `predict_model` function returns `Cluster` label as a new column in the input dataframe. This step may or may not be needed depending on the use-case. Some times clustering models are trained for analysis purpose only and the interest of user is only in assigned labels on the training dataset, that can be done using `assign_model` function. `predict_model` is only useful when you want to obtain cluster labels on unseen data (i.e. data that was not used during training the model).

# In[20]:


# predict on test set
kmeans_pred = predict_model(kmeans, data=data)
kmeans_pred


# ## Save Model

# Finally, you can save the entire pipeline on disk for later use, using pycaret's `save_model` function.

# In[19]:


# save pipeline
save_model(kmeans, './Output/Model_kmeans')


# In[21]:


# load pipeline
kmeans_pipeline = load_model('./Output/Model_kmeans')
kmeans_pipeline


# In[ ]:




