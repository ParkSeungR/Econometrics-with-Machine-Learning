#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[2]:


# 라이브러리 불러오기
exec(open('Functions/Machine Learning_Econometrics_Lib.py').read())


# # 분류문제

# In[3]:


# 데이터 불러오기
filename = "Data/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)


# In[4]:


from pycaret.classification import *
s = setup(data, target = 'class', session_id = 12345)


# In[5]:


best = compare_models()


# In[6]:


print(best)


# In[6]:


evaluate_model(best)


# In[13]:


plot_model(best, plot = 'auc')


# In[14]:


plot_model(best, plot = 'confusion_matrix')


# In[15]:


predict_model(best)


# In[16]:


predictions = predict_model(best, data=data)
predictions.head()


# In[17]:


predictions = predict_model(best, data=data, raw_score=True)
predictions.head()


# In[18]:


save_model(best, 'Model/my_best_pipeline')


# In[19]:


loaded_model = load_model('Model/my_best_pipeline')
print(loaded_model)


# In[ ]:





# # 회귀분석

# In[22]:


# 데이터세트 불러오기
filename = 'Data/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(filename, delim_whitespace=True, names=names)


# In[23]:


from pycaret.regression import *
s = setup(data, target = 'MEDV', session_id = 12345)


# In[24]:


best = compare_models()


# In[25]:


print(best)


# In[26]:


evaluate_model(best)


# In[27]:


plot_model(best, plot = 'residuals')


# In[28]:


plot_model(best, plot = 'feature')


# In[29]:


predict_model(best)


# In[30]:


predictions = predict_model(best, data=data)
predictions.head()


# In[31]:


save_model(best, 'Model/my_best_pipeline')


# In[32]:


loaded_model = load_model('Model/my_best_pipeline')
print(loaded_model)


# In[ ]:





# In[33]:


# load sample dataset
from pycaret.datasets import get_data
data = get_data('jewellery')


# In[37]:


# 데이터세트 불러오기
data = read_csv('Data/jewellery.csv')
data


# In[38]:


from pycaret.clustering import *
s = setup(data, normalize = True)


# In[41]:


kmeans = create_model('kmeans')


# In[42]:


print(kmeans)


# In[43]:


evaluate_model(kmeans)


# In[44]:


plot_model(kmeans, plot = 'elbow')


# In[45]:


plot_model(kmeans, plot = 'silhouette')


# In[46]:


result = assign_model(kmeans)
result.head()


# In[47]:


predictions = predict_model(kmeans, data = data)
predictions.head()


# In[48]:


save_model(kmeans, 'Model/kmeans_pipeline')


# In[49]:


loaded_model = load_model('Model/kmeans_pipeline')
print(loaded_model)


# In[ ]:





# In[50]:


from pycaret.datasets import get_data
data = get_data('airline')


# In[51]:


from pycaret.time_series import *
s = setup(data, fh = 3, fold = 5, session_id = 12345)


# In[52]:


best = compare_models()


# In[53]:


plot_model(best, plot = 'forecast', data_kwargs = {'fh' : 24})


# In[54]:


plot_model(best, plot = 'diagnostics')


# In[55]:


plot_model(best, plot = 'insample')


# In[56]:


final_best = finalize_model(best)
predict_model(best, fh = 24)


# In[57]:


save_model(final_best, 'Model/my_final_best_model')


# In[58]:


loaded_model = load_model('Model/my_final_best_model')
print(loaded_model)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




