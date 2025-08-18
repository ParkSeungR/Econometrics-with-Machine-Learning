#!/usr/bin/env python
# coding: utf-8

# # 14. Time Series Models of Heteroscedasticity

# In[3]:


pip install arch


# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[2]:


# 본서에서 사용되는 모든 Library 불러오기 
exec(open('Functions/Traditional_Econometrics_Lib.py').read())


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm  
import arch
import warnings  
get_ipython().run_line_magic('matplotlib', 'inline')

#Ignoring the warnings  
warnings.filterwarnings("ignore")  


# In[4]:


#그래프 형태와 크기 정의(Graphics format & Size) 
#그래프 그리기 위한 라이브러리는 주로 seaborn 사용
sns.set_style('darkgrid')  
pd.plotting.register_matplotlib_converters()  
sns.mpl.rc('figure',figsize=(12, 8))  

# 데이터의 범위 지정  
start = '2010-01-01'  
end   = '2024-01-31'  

# 불러올 주가(코스피, S&P, 나스닥, 니케이)  
assets = ['^KS11', '^GSPC', '^IXIC', '^N225']  
#assets.sort()  

#Downloading price data  
data = yf.download(assets, start=start, end=end)  
data = data.loc[:, ('Adj Close', slice(None))]  
data.columns = assets  
data.columns = ['KOSPI','SNP500', 'NASDAK', 'NIKKEI']


# In[4]:


data


# In[5]:


# 주가지수 그래프 그리기
fig, ax = plt.subplots(4, 1)  
ax = np.ravel(ax)  
ax[0].plot(data['KOSPI'])  
ax[1].plot(data['SNP500'])  
ax[2].plot(data['NASDAK'])
ax[3].plot(data['NIKKEI']) 

ax[0].set_title('코스피(KOSPI Composite Index)')  
ax[1].set_title('스탠다드 앤 푸어스(S&P 500 Index)')  
ax[2].set_title('나스닥(NASDAQ Composite Index)')  
ax[3].set_title('니케이(NEKKEI Composite Index)')  
fig.tight_layout()
plt.show() 


# In[6]:


#일별자료를 주별자료 변환하려면 resample 사용  
data_week = data.resample('W').last()  
data_week 


# In[7]:


# 주가지수 그래프 그리기
fig, ax = plt.subplots(4, 1)  
ax = np.ravel(ax)  
ax[0].plot(data_week['KOSPI'])  
ax[1].plot(data_week['SNP500'])  
ax[2].plot(data_week['NASDAK'])
ax[3].plot(data_week['NIKKEI']) 

ax[0].set_title('코스피(KOSPI Composite Index)')  
ax[1].set_title('스탠다드 앤 푸어스(S&P 500 Index)')  
ax[2].set_title('나스닥(NASDAQ Composite Index)')  
ax[3].set_title('니케이(NEKKEI Composite Index)')  
fig.tight_layout()
plt.show() 


# In[8]:


#일별자료로 수익률 계산
returns = data.pct_change().dropna() * 100  
display(returns) 


# In[9]:


# 주가지수 수익률 그래프 그리기
fig, ax = plt.subplots(4, 1)  
ax = np.ravel(ax)  
ax[0].plot(returns['KOSPI'])  
ax[1].plot(returns['SNP500'])  
ax[2].plot(returns['NASDAK'])
ax[3].plot(returns['NIKKEI']) 

ax[0].set_title('코스피(KOSPI Composite Returns)')  
ax[1].set_title('스탠다드 앤 푸어스(S&P 500 returns)')  
ax[2].set_title('나스닥(NASDAQ Composite Returns)')  
ax[3].set_title('니케이(NEKKEI Composite Returns)')  
fig.tight_layout()
plt.show() 


# In[7]:


# 주가지수 수익률 그래프 그리기
fig, ax = plt.subplots(1, 2, figsize=(12,3))  
ax[0].set_title('코스피(KOSPI Composite Returns)')  
ax[0].plot(returns['KOSPI']) 
sns.distplot(returns['KOSPI'], ax=ax[1], hist=True, kde=True, fit=stats.norm, bins=30,
           color='0.4', hist_kws={'edgecolor':'white'},
           kde_kws={'linewidth': 2, 'linestyle': '--'})
fig.tight_layout()
plt.show() 


# In[10]:


# 주가지수 수익률 그래프 그리기
fig, ax = plt.subplots(1, 2, figsize=(12,3))  
ax[0].set_title('스탠다드 앤 푸어스(S&P 500 returns)')  
ax[0].plot(returns['SNP500']) 
sns.distplot(returns['SNP500'], ax=ax[1], hist=True, kde=True, fit=stats.norm, bins=30,
           color='0.4', hist_kws={'edgecolor':'white'},
           kde_kws={'linewidth': 2, 'linestyle': '--'})
fig.tight_layout()
plt.show() 


# In[11]:


# ARCH(1) volatility의 AR(1) 모형 추정 

y = returns['KOSPI']  
x = returns['SNP500']
model = arch.arch_model(y=y,x=x, mean='ARX', vol='arch', p=1)  
res = model.fit(disp='off')  
print(res.summary()) 
 


# In[18]:


res.arch_lm_test(lags=1, standardized= False)


# In[13]:


# ARCH(4) volatility의 AR(4) 모형 추정 

y = returns['KOSPI']  
x = returns['SNP500']
model = arch.arch_model(y=y, x=x, mean='ARX', vol='arch', p=4)  
res = model.fit(disp='off')  
print(res.summary()) 
 


# In[23]:


# standardized residuals와 conditional volatility 그리기
fig = res.plot() 


# In[10]:


# ARCH(1) volatility의 AR(1) 모형 추정 

y = returns['KOSPI'] 
x = returns['SNP500']
model = arch.arch_model(y=y, mean='AR', lags=[1], vol='arch', p=1)  
res = model.fit(disp='off')  
display(res.summary()) 
 


# In[22]:


res.arch_lm_test(lags=4, standardized= False)


# In[11]:


# S&P주가 수익율의 코스피 주가수익율에 미치는 영향에 대한 AIC 통계기반의 AR-X(p) 모형  
x = returns['SNP500'] 

# 모형정의   
sel = ar_select_order(y, maxlag=8, ic='aic', glob=True, trend='c', exog=x)  

# 선정기준 통계  
#'aic': Akaike Information Criterion  
#'bic': Bayesian or Schwarz Information Criterion  
#'hqic': Hannan Quinn Information Criterion  

# 최상의 모형 선정  
res = sel.model.fit()  
 
display(res.summary()) 


# In[12]:


# S&P주가 수익율의 코스피 주가수익율에 미치는 영향에 대한 AIC 통계기반의 AR-X(p) 모형  
x = returns['SNP500'] 

# 모형정의   
model = arch.arch_model(y=y, mean='AR', lags=[2,4,6,7,8], vol='arch', p=7)  

# 선정기준 통계  
#'aic': Akaike Information Criterion  
#'bic': Bayesian or Schwarz Information Criterion  
#'hqic': Hannan Quinn Information Criterion  

# 최상의 모형 선정  
res = model.fit()  
 
display(res.summary()) 


# In[13]:


res.arch_lm_test(lags=4, standardized= False)


# In[14]:


# LM Heteroscedasticity Test 
# 오차항의 제곱과 그 시차변수  
q = 4  
e2 = res.resid.values**2  
n = len(e2)  
e2 = e2.reshape(n, 1)  
X = np.zeros((n-q, 0)) 

for i in range(0, q):  
    X = np.concatenate([X, e2[i:n-q+i,:]], axis=1) 
print(X)

e2 = e2[q:]  
print(e2)

labels = ['LM-stat', 'LM: p-value', 'F-value', 'F: p-value']  
BP = dg.het_breuschpagan(resid=e2, exog_het = X)  
display(pd.DataFrame(BP, index=labels, columns=['Value']).applymap("{:.4f}".format)) 


# In[51]:


# 최상의 ARCH(q) 모형 선택  
 
model = arch.arch_model(y=y, x=x, mean='ARX', lags=[7], vol='arch', p=8)  
res = model.fit(disp='off')  
display(res.summary()) 


# In[53]:


# standardized residuals와 conditional volatility 그리기
fig = res.plot() 


# ### 14.3 GARCH Model

# In[15]:


# 코스피 지수에 대한 GARCH(1,1) volatility인 AR(1) 모형 추정
model = arch.arch_model(y=y, mean='AR', lags=[1], vol='garch', p=1, q=1)  
res = model.fit(disp='off')  
print(res.summary()) 


# In[24]:


# 코스피 지수에 대한 GARCH(1,1) volatility인 AR(1) 모형 추정
model = arch.arch_model(y=y, x=x, mean='ARX', vol='garch', p=1, q=1)  
res = model.fit(disp='off')  
display(res.summary()) 


# In[26]:


#Graphing the standardized residuals and conditional volatility  
fig = res.plot() 


# In[54]:


# Estimating an AR-X(p) model with volatility GARCH (1,1)  #for the KOSPI Composite Index  
model = arch.arch_model(y=y,  x=x, mean='ARX', lags=[7], vol='garch', p=1, q=2)  
res = model.fit(disp='off') 
display(res.summary()) 


# In[55]:


# Graphing the standardized residuals  and conditional volatility 
fig = res.plot()


# In[ ]:





# In[27]:


# Selecting the best AIC-based AR-X(p) model for the KOSPI Composite Index  

# Creating the model set  
sel = ar_select_order(y, maxlag=8, ic='aic', glob=True, trend='c', exog=x)  

# Selecting the best model  
res = sel.model.fit()  
res.summary()  
display(res.summary()) 


# In[28]:


# Graphing the ACF and PCAF functions  
fig, ax = plt.subplots(2, 1)  
ax = np.ravel(ax) 
sm.graphics.tsa.plot_acf(res.resid**2, lags=40, ax=ax[0], zero=False)  
sm.graphics.tsa.plot_pacf(res.resid**2, lags=40, ax=ax[1], zero=False)  
fig.tight_layout()
plt.show() 


# In[29]:


# Ljung-Box Test 
e2 = res.resid.values**2  
Q = sm.stats.acorr_ljungbox(e2, lags=range(1, 6), return_df=True)  
display(Q.applymap("{:.4f}".format)) 


# In[30]:


# Estimating an AR-X(p) model with volatility GARCH (1,1)  #for the KOSPI Composite Index  
model = arch.arch_model(y=y,  x=x, mean='ARX', lags=[5], vol='garch', p=1, q=2)  
res = model.fit(disp='off') 
display(res.summary()) 


# In[31]:


# Graphing the standardized residuals  and conditional volatility 
fig = res.plot() 


# In[42]:


# Estimating an AR-X(p) model with volatility GARCH (1,1)  #for the KOSPI Composite Index  
model = arch.arch_model(y=y,  x=x, mean='ARX', vol='garch', p=1, q=2)  
res = model.fit(disp='off') 
display(res.summary()) 


# In[43]:


fig = res.plot() 


# In[44]:


am = arch_model(y=y, x=x, mean='ARX', p=1, q=2)
res = am.fit()
print(res.summary())


# In[45]:


fig = res.plot() 


# In[46]:


am = arch_model(y=y, x=x, mean='ARX', p=1, o=1, q=1)
res = am.fit()
print(res.summary())


# In[47]:


fig = res.plot() 


# In[50]:


am = arch_model(y=y, x=x, mean='ARX', vol='egarch', p=1, o=1, q=1)
res = am.fit()
print(res.summary())


# In[38]:


am = arch_model(y=y, x=x, mean='ARX', p=1, o=1, q=1, power=1.0)
res = am.fit()
print(res.summary())


# In[ ]:





# In[52]:


# 모형의 차수선정
sel = ar_select_order(y, maxlag=8, ic='aic', glob=True, trend='c', exog=x)  
res = sel.model.fit()  
res.summary()


# In[53]:


model = arch_model(y=y, x=x, mean='ARX', lags=[7], vol='arch', p=8)  
res = model.fit()  
display(res.summary()) 


# In[54]:


fig = res.plot() 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




