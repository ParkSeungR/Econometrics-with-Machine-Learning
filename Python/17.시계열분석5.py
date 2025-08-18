#!/usr/bin/env python
# coding: utf-8

# # Cointegration and VECM

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[2]:


exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[4]:


# 한국의 거시경제 통계자료 불러오기
data = pd.read_csv('Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# In[5]:


df = data[[ 'gdp', 'inv', 'con']]
df


# In[6]:


df.info()


# In[7]:


df.describe().transpose()


# In[8]:


# Graphics size
sns.mpl.rc('figure', figsize=(12, 3))
plt.plot(df)


# ##  공적분(Cointegration)

# In[9]:


# 가성회귀(Spurious regression)
res = smf.ols('con ~ gdp', data=data).fit()
print(res.summary())


# In[10]:


# 가성회귀 잔차의 그래프
res.resid.plot()


# In[13]:


# Augmented Dickey-Fuller Test
for i in ['n','c','ct', 'ctt']:    
    adf_tt = ADF(df['gdp'], trend=i, lags=4)  
    print(adf_tt.summary().as_text()) 


# In[14]:


# Augmented Dickey-Fuller Test
for i in ['n','c','ct', 'ctt']:    
    adf_tt = ADF(df['con'], trend=i, lags=4)  
    print(adf_tt.summary().as_text()) 


# In[15]:


# Augmented Dickey-Fuller Test
for i in ['n','c','ct', 'ctt']:    
    adf_tt = ADF(df['inv'], trend=i, lags=4)  
    print(adf_tt.summary().as_text()) 


# In[16]:


# 공적분된 2변수의 VECM추정
y = np.log(data['con'])
X = np.log(data['gdp'])
endog = pd.concat([y, X], axis=1)

model = VECM(endog=endog, exog=None, exog_coint=None, k_ar_diff=2, coint_rank=1, deterministic='ct')
vecm_res = model.fit()

print(vecm_res.summary())


# In[17]:


eg_test = engle_granger(y, X, trend="n")
print(eg_test)


# ## 16.2.3	Vector Error Correction Model (VECM)

# In[18]:


# The VECM model with realcons and realgdp
from statsmodels.tsa.vector_ar.vecm import VECM
endog = pd.concat([y, X], axis=1)

# deterministic can assume the following values:
# 'nc' : no deterministic terms
# 'co' : constant outside the cointegration relationship
# 'ci' : constant within the cointegration relationship
# 'lo' : linear trend outside the cointegration relationship
#‘li’ : linear trend within the cointegration relationship

model = VECM(endog=endog, exog=None, exog_coint=None, k_ar_diff=2, coint_rank=1, deterministic='nc')
vecm_res = model.fit()

print(vecm_res.summary())


# ## 16.2.4	Johansen Cointegration Test

# In[22]:


# Making the Johansen test  
from statsmodels.tsa.vector_ar.vecm import select_coint_rank  

# det_order can assume the following values:  
# -1: No trend  
# 0: With constant  
# 1: With linear trend  

# method can assume the following values:  
# 'trace': trace test 
# 'maxeig': maximum eigenvalue test  

rank_test = select_coint_rank(endog=endog, det_order=-1, k_ar_diff=2, method="maxeig", signif=0.05)  

# Rank of Pi matrix  
print('Rank of Pi matrix: ', rank_test.rank)  

# Johansen test result  
print(rank_test.summary()) 


# ### 16.2.5	Johansen Methodology

# In[23]:


# Graphing the model variables
fig, ax = plt.subplots()
ax.plot(y, label='con')
ax.plot(X, label='gdp')
ax.legend(loc='best')

plt.plot()


# In[24]:


#Selection of the order of the residuals  

from statsmodels.tsa.vector_ar.vecm import select_order 

lag_order = select_order(data=endog, maxlags=10, deterministic='co', seasons=0)  

print(lag_order.summary()) 


# In[25]:


# Lags selected by indicator  
print('Akaike Information Criterion :',lag_order.aic)  
print('Schwarz Information Criterion :',lag_order.bic)  
print('Final Prediction Error :',lag_order.fpe)  
print('Hannan-Quinn Criterion :',lag_order.hqic)  


# In[26]:


# Determining the rank of Pi matrix  

# det_order can assume the following values:  
# -1: No trend  
# 0: With constant  
# 1: With linear trend  

# method can assume the following values:  
# 'trace': trace test 
# 'maxeig': maximum eigenvalue test  

rank_test = select_coint_rank(endog=endog, det_order=-1, k_ar_diff=lag_order.aic,  method="trace",  signif=0.05) 

# Rank of Pi matrix  
print('Rank of Pi matrix: ', rank_test.rank)  



# In[27]:


# Estimating the VECM model  
model = VECM(endog=endog, exog=None, exog_coint=None, k_ar_diff=lag_order.aic, coint_rank=rank_test.rank, deterministic='co')  

vecm_res = model.fit()  
print(vecm_res.summary()) 


# In[28]:


# Graphing Impulse Response Functions
fig = vecm_res.irf(10).plot() 



# In[29]:


# 예측
forecast = vecm_res.predict(10)  
forecast = pd.DataFrame(forecast, columns=endog.columns)  
display(forecast) 


# In[43]:


# Graphing the forecast of the variables  
vecm_res.plot_forecast(10) 


# In[ ]:





# In[ ]:




