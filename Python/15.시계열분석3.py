#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[2]:


# 본서에서 사용되는 모든 Library 불러오기 
exec(open('Functions/Traditional_Econometrics_Lib.py').read())


# In[3]:


# 한국의 거시경제 통계자료 불러오기
data = pd.read_csv('Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# In[10]:


df = data[[ 'gdp', 'inv', 'con']]
df


# In[11]:


# 퍼센트 변화(percent change) 
df_gr = 100*df.pct_change(periods=4).dropna()
df_gr = df_gr.loc['1970-01-31':]
print(df_gr)


# In[12]:


df_gr.plot(figsize=(10, 4));


# In[13]:


for i in range(1, 9):
    model = VAR(df_gr)
    results = model.fit(i)
    print('Order =', i)
    print('AIC: ', results.aic)
    print('BIC: ', results.bic)
    print()


# In[14]:


model.select_order(8)
results = model.fit(maxlags=8, ic='aic')
lag_order = results.k_ar
print(lag_order)


# In[15]:


model = VAR(df_gr)
results = model.fit(5)
results.summary()


# In[16]:


results.plot();


# In[17]:


results.plot_acorr();


# In[18]:


results.forecast(df_gr.values[lag_order:], 20)


# In[19]:


results.plot_forecast(20);


# In[20]:


irf = results.irf(10)  
irf.plot(orth=False);


# In[21]:


irf = results.irf(10)  
irf.plot(orth=True);


# In[22]:


irf.plot(impulse='gdp', );


# In[94]:


irf.plot(impulse='gdp', orth=True);


# In[23]:


irf.plot_cum_effects(orth=False);


# In[24]:


irf.plot_cum_effects(orth=True);


# In[25]:


fevd = results.fevd(10)
fevd.summary() 


# In[164]:


results.fevd(20).plot();
    


# In[26]:


cau_gdp = results.test_causality('gdp', ['inv', 'con'], kind='f')
cau_gdp.summary()


# In[27]:


nor = results.test_normality()
nor.summary()


# ### SVAR

# In[28]:


#define structural inputs
A = np.asarray([[1, 0, 0],['E', 1, 0],['E', 'E', 1]])
B = np.asarray([['E', 0, 0], [0, 'E', 0], [0, 0, 'E']])
A_guess = np.asarray([0.5, 0.25, -0.38])
B_guess = np.asarray([0.5, 0.1, 0.05])
mymodel = SVAR(df_gr, svar_type='AB', A=A, B=B, freq='Q')
res = mymodel.fit(maxlags=3, maxiter=10000, maxfun=10000, solver='bfgs')


# In[29]:


fevd = res.fevd(10)
fevd.summary()


# In[30]:


res.fevd(20).plot()


# In[31]:


res.irf(periods=20).plot(impulse='gdp', plot_stderr=True,
                         stderr_type='mc', repl=100);


# # VARMAX

# In[7]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()

# 본서에서 사용되는 모든 Library 불러오기 
exec(open('Functions/Traditional_Econometrics_Lib.py').read())


# In[8]:


# 한국의 거시경제 통계자료 불러오기
data = pd.read_csv('Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index

df = data[[ 'gdp', 'inv', 'con', 'gov']]
df


# In[9]:


# 퍼센트 변화(percent change) 
df_gr = 100*df.pct_change(periods=4).dropna()
df_gr = df_gr.loc['1970-01-31':]
print(df_gr)


# In[10]:


endog = df_gr.loc[:, ['gdp', 'inv', 'con']]
exog = df_gr.loc[:, 'gov']


# In[ ]:


# VARX 
Model_VAR = VARMAX(endog=endog, order=(5,0), exog=exog)
Results_VAR = Model_VAR.fit(maxiter=1000, disp=False)


# In[12]:


print(Results_VAR.summary())


# In[18]:


ax = Results_VAR.impulse_responses(10, orthogonalized=True, impulse=[1, 0, 0]).plot(figsize=(12,4))
ax.set(xlabel='t', title='Responses to a shock to `gdp`');


# In[14]:


ax = Results_VAR.impulse_responses(10, orthogonalized=True, impulse=[0, 1, 0]).plot(figsize=(12,4))
ax.set(xlabel='t', title='Responses to a shock to `inv`');


# In[16]:


ax = Results_VAR.impulse_responses(10, orthogonalized=True, impulse=[0, 0, 1]).plot(figsize=(12,4))
ax.set(xlabel='t', title='Responses to a shock to `con`');


# In[19]:


# VMA 
Model_VMA = VARMAX(endog, order=(0, 2), error_cov_type='diagonal')
Results_VMA = Model_VMA.fit(maxiter=1000, disp=False)
print(Results_VMA.summary())


# In[20]:


# VARMA(p,q)
Model_VARMA = VARMAX(endog, order=(2,2))
Results_VARMA = Model_VARMA.fit(maxiter=1000, disp=False)
print(Results_VARMA.summary())

