#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[4]:


exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[8]:


import wooldridge as woo
import pandas as pd
import linearmodels as plm

wagepan = woo.dataWoo('wagepan')
wagepan = wagepan.set_index(['nr', 'year'], drop=False)


# In[9]:


df = wagepan[['nr', 'year', 'lwage', 'educ', 'exper', 'hours', 'married', 'union', 'south', 'hisp', 'black']]
print(df)


# In[11]:


# 기초 통계
df_descID = df.groupby(level=0, axis=0).describe().T
print(df_descID)


# In[13]:


df_descYR = df.groupby(level=1, axis=0).describe().T
print(df_descYR)


# # 1. 혼용회귀(Pooled Regression)
# ## 1) OLS
# ## 2) PooledOLS
# ## 3) Between OLS
# ## 4) First Difference
# ## 5) Within OLS

# In[29]:


# OLS
df['exper2'] = df['exper']**2
y = df['lwage']
X = df[['educ', 'exper', 'exper2', 'hours', 'married', 'union', 'south', 'black']]
X = sm.add_constant(X)
Results_OLS = sm.OLS(y, X).fit()
print(Results_OLS.summary())


# In[21]:


# Pooled OLS
Results_POOL = lm.panel.PooledOLS(y, X).fit() 
print(Results_POOL)


# In[22]:


# BetweenOLS Estimation
Results_BETWEEN = lm.panel.BetweenOLS(y, X).fit() 
print(Results_BETWEEN)


# In[38]:


# FirstDifferenceOLS Estimation: 개체내 불변변수, 상수항 제외되어야
y = df['lwage']
X = df[['exper', 'exper2', 'hours', 'married', 'union', 'south']]
Results_FD = lm.panel.FirstDifferenceOLS(y, X).fit() 
print(Results_FD)


# In[39]:


# 개체내 평균과의 차이: FD의 독립변수와 동일
y_w = y - y.groupby(level=0, axis=0).transform('mean')
X_w = X - X.groupby(level=0, axis=0).transform('mean')

Results_WITHIN = lm.panel.PooledOLS(y_w, X_w).fit() 
print(Results_WITHIN) 


# # 2. 고정효과모형(Fixed Effects(FE)
# ## 1) One-way Fixed Effects
# ## 2) Two-way Fixed Effects
# 

# In[46]:


# One-way Fixed Effects
df['exper2'] = df['exper']**2
y = df['lwage']
X = df[['exper', 'exper2', 'hours', 'married', 'union', 'south']]
X = sm.add_constant(X)

Results_FE1 = lm.panel.PanelOLS(y, X, entity_effects=True).fit()  
print(Results_FE1) 
print(Results_FE1.estimated_effects.head(10))


# In[51]:


y = df['lwage']
X = df[['hours', 'married', 'union', 'south']]

# Two-way Fixed Effects
Results_FE2 = lm.panel.PanelOLS(y, X, entity_effects=True, time_effects=True).fit()  
print(Results_FE2) 
print(Results_FE2.estimated_effects.head(10))


# # 3. 확률효과모형(Random Effects(RE)

# In[54]:


# Random Effects(RE)
df['exper2'] = df['exper']**2
y = df['lwage']
X = df[['educ', 'exper', 'exper2', 'hours', 'married', 'union', 'south', 'black']]
X = sm.add_constant(X)
Results_RE = lm.panel.RandomEffects(y, X).fit()  
print(Results_RE) 


# # 4. 파라미터의 공분산 추정치(강건 표준오차)

# In[62]:


# Robust Standard errors using OLS and Linearmodels
# 혼용회귀
df['exper2'] = df['exper']**2
y = df['lwage']
X = df[['educ', 'exper', 'exper2', 'hours', 'married', 'union', 'south', 'black']]
X = sm.add_constant(X)

# 단순 OLS 
Results_pooled_ols = lm.panel.PooledOLS(y, X).fit()

# 이분산 문제(Using White standard errors)
Results_pooled_hec = lm.panel.PooledOLS(y, X).fit(cov_type='heteroskedastic')

# 자기상관 문제(Using Cluster standard errors)
Results_pooled_clu = lm.panel.PooledOLS(y, X).fit(cov_type='clustered', cluster_entity=True)

# 이분산 & 자기상관 문제(Using Driscoll Kraay standard errors)
Results_pooled_dk = lm.panel.PooledOLS(y, X).fit(cov_type='kernel')

# Comparing standard errors and confidence intervals 
print(Results_pooled_ols.summary.tables[1]) 
print(Results_pooled_hec.summary.tables[1]) 
print(Results_pooled_clu.summary.tables[1]) 
print(Results_pooled_dk.summary.tables[1])


# In[64]:


# 고정효과 모형(One-way Fixed Effects)
df['exper2'] = df['exper']**2
y = df['lwage']
X = df[['exper', 'exper2', 'hours', 'married', 'union', 'south']]
X = sm.add_constant(X)

# 단순 OLS 
Results_FE1_ols = lm.panel.PanelOLS(y, X).fit()

# 이분산 문제(Using White standard errors)
Results_FE1_hec = lm.panel.PanelOLS(y, X).fit(cov_type='heteroskedastic')

# 자기상관 문제(Using Cluster standard errors)
Results_FE1_clu = lm.panel.PanelOLS(y, X).fit(cov_type='clustered', cluster_entity=True)

# 이분산 & 자기상관 문제(Using Driscoll Kraay standard errors)
Results_FE1_dk = lm.panel.PanelOLS(y, X).fit(cov_type='kernel')

# Comparing standard errors and confidence intervals 
print(Results_FE1_ols.summary.tables[1]) 
print(Results_FE1_hec.summary.tables[1]) 
print(Results_FE1_clu.summary.tables[1]) 
print(Results_FE1_dk.summary.tables[1])


# In[66]:


# 확률효과 모형(Random Effects)
df['exper2'] = df['exper']**2
y = df['lwage']
X = df[['educ', 'exper', 'exper2', 'hours', 'married', 'union', 'south', 'black']]
X = sm.add_constant(X)

# 단순 OLS 
Results_RE_ols = lm.panel.RandomEffects(y, X).fit()

# 이분산 문제(Using White standard errors)
Results_RE_hec = lm.panel.RandomEffects(y, X).fit(cov_type='heteroskedastic')

# 자기상관 문제(Using Cluster standard errors)
Results_RE_clu = lm.panel.RandomEffects(y, X).fit(cov_type='clustered', cluster_entity=True)

# 이분산 & 자기상관 문제(Using Driscoll Kraay standard errors)
Results_RE_dk = lm.panel.RandomEffects(y, X).fit(cov_type='kernel')

# Comparing standard errors and confidence intervals 
print(Results_RE_ols.summary.tables[1]) 
print(Results_RE_hec.summary.tables[1]) 
print(Results_RE_clu.summary.tables[1]) 
print(Results_RE_dk.summary.tables[1])


# # 하우스만 검정(Hausman test)

# In[12]:


import wooldridge as woo
import numpy as np
import linearmodels as plm
import scipy.stats as stats

wagepan = woo.dataWoo('wagepan')
wagepan = wagepan.set_index(['nr', 'year'], drop=False)

# FE와 RE 추정
Results_FE = plm.PanelOLS.from_formula(formula='lwage ~ I(exper**2) + married +'
                                           'union + C(year) + EntityEffects',
                                       data=wagepan).fit()
print(Results_FE)
Results_RE = plm.RandomEffects.from_formula(
            formula='lwage ~ educ + black + hisp + exper + I(exper**2)'
                         '+ married + union + C(year)', data=wagepan).fit()
print(Results_RE)

# (1) 공통 파라미터 정의 
C_coef = list(set(Results_FE.params.index).intersection(Results_RE.params.index))
# (2) FE and RE 파라미터 추정치의 차이
D_b = np.array(Results_FE.params[C_coef] - Results_RE.params[C_coef])
dof = len(D_b)
D_b.reshape((dof, 1))
# (3) FE and RE 파라미터 공분산 행렬의 차이
D_cov = np.array(Results_FE.cov.loc[C_coef, C_coef] -
                 Results_RE.cov.loc[C_coef, C_coef])
D_cov.reshape((dof, dof))
# (4) Chi2 통계량 계산
Chi2 = abs(np.transpose(D_b) @ np.linalg.inv(D_cov) @ D_b)
pvalue = 1 - stats.chi2.cdf(Chi2, dof)

print(Chi2, dof, pvalue)

# 함수이용
exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/HausmanTest.py').read())
H, dof, pvalue = HausmanTest(Results_FE, Results_RE)
print(H, dof, pvalue)


# # Mundlak test

# In[12]:


import wooldridge as woo
import numpy as np
import pandas as pd
import linearmodels as lm
import scipy.stats as stats

wagepan = woo.dataWoo('wagepan')
#wagepan = wagepan.set_index(['nr', 'year'], drop=True)
print(wagepan) 


# In[13]:


df_m = wagepan.groupby('nr').mean()                         
df = wagepan.merge(df_m, on = 'nr', how = 'left')      
print(df)
df.info()

df = df.set_index(['nr', 'year_x'], drop=True) 
print(df)


# In[14]:


formula = """lwage_x ~ 1 + educ_x + exper_x + expersq_x + 
             black_x + south_x + union_x + exper_y + expersq_y + 
             south_y + union_y""" 

hnull = ['exper_y = 0.0', 'expersq_y = 0.0', 'south_y = 0.0', 'union_y = 0.0']
               
# 확률효과 모형
Results_RE = lm.RandomEffects.from_formula(formula, df).fit()  
# Mundlak stat. 
Mundlak_RE = Results_RE.wald_test(formula = hnull)   
print(Mundlak_RE)

# 확률효과 모형(Robust)
Results_RECL = lm.RandomEffects.from_formula(formula, df).fit(cov_type = 'clustered')

# Mundlak stat.
Mundlak_RECL = Results_RECL.wald_test(formula = hnull)     
print(Mundlak_RECL)


# In[15]:


# Fixed-effects model
Results_FE = lm.panel.PanelOLS.from_formula(formula + ' + EntityEffects', 
                                            df, drop_absorbed = True).fit()


# In[17]:


# Fixed-effects model
m4f = plm.panel.PanelOLS.from_formula(formula + ' + EntityEffects', 
                                            df, drop_absorbed = True).fit(cov_type = 'clustered', cluster_entity = True)


# In[18]:


# Mundlak statistics 
tbl = pd.concat([m2f.params, m2f.std_errors, m3f.std_errors, m4f.params, m4f.std_errors], axis =1) 

tbl.columns = ['coef.RE', 'se.RE', 'se.RE cluster', 'coef.FE', 'se.FE cluster']
tbl.loc['Mundlak'] = ['', Mstat2, Mstat3, '', ''] 
round(tbl.fillna(''), 4)


# # Hausman Taylor

# In[47]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()

import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from statsmodels.sandbox.regression.gmm import IV2SLS


# In[137]:


df = pd.read_csv('Data/psidextract.csv', parse_dates = ['t'], index_col = ['id', 't'])
display(df)


# In[2]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from statsmodels.sandbox.regression.gmm import IV2SLS


# In[3]:


# Read CSV file
data = pd.read_csv('Data/psidextract.csv', parse_dates=['t'], index_col=['id', 't'])
print(data)

# Define variables
y  = data['lwage']
X1 = data[["occ", "south", "smsa", "ind"]]
X2 = data[["exp", "exp2", "wks", "ms", "union"]]
Z1 = data[["fem", "blk"]]
Z2 = data["ed"]
print(X1, X2, Z1, Z2)


# In[4]:


# #####################################################
# Time length 
T = len(y.index.get_level_values(1).unique())
print(T)

# 1) Step 1: Fixed effects equation
X = pd.concat([X1, X2], axis=1)
XF = sm.add_constant(X)
print(XF)

# PanelOLS requires entity (individual) and time effects to be specified if needed
Model_FE = PanelOLS(y, XF, entity_effects=True)
Results_FE = Model_FE.fit()
print(Results_FE)


# In[5]:


# Step 2 
# Residuals
FE_resid = Results_FE.resids
print(FE_resid)
sig1 = Results_FE.s2
print(sig1)

ZZ1 = sm.add_constant(Z1)
Z = pd.concat([ZZ1, Z2], axis=1)
INST = pd.concat([X1, ZZ1], axis=1)
print(ZZ1, Z2, INST)

# IV2SLS requires dependent variable, exogenous regressors, endogenous regressors, and instruments
Results_IV = IV2SLS(FE_resid, Z, INST).fit()
print(Results_IV.summary())
sig2 = Results_IV.scale
print(sig2)


# In[6]:


# 3) Step 3
ahat = 1 - np.sqrt(sig1/(T * sig2 + sig1))
df = pd.concat([y, X1, X2, Z1, Z2], axis=1)

# 원자료(_x), ID별 평균(_y), 평균에서의 편차(_d), 가중평균에서의 편차(_s)
# 1) 평균값: _y 
df_m = df.groupby('id').mean()
df_m = pd.merge(df, df_m, on = 'id', how = 'left')

# 2) 평균에서의 차이(xy - xy_bar): _d 
for name in df.columns:
    df_m[name +'_d'] = df_m[name + '_x'] - df_m[name + '_y']

# 3) 가중 평균에서의 차이((xy - ahat * xy_bar): _s
for name in df.columns:                          
    df_m[name +'_s'] = df_m[name + '_x'] - ahat * df_m[name + '_y']

print(df_m)
print(df_m.describe().T)


# In[7]:


# 가중평균에서의 편차 변수 (_s)
# define dep, indep var.
df_s = df_m[[var for var in df_m if var.endswith('_s')]]
y_s = df_m.iloc[:,0]
X_s = df_m.iloc[:,1:]
X_s = sm.add_constant(X_s)
print(df_s, y_s, X_s)
print(X_s.describe().T)


# In[9]:


# 대변수 정의 (_d, _y, Z1)
ZZ_d = df_m[[var for var in df_m if var.endswith('_d')]]
ZZ_d = ZZ_d.iloc[:,1:]
ZZ_dd= ZZ_d.reset_index()
ZZ_dd = ZZ_dd.iloc[:,1:]
print(ZZ_dd)
      
ZZ_y = df_m[[var for var in df_m if var.endswith('_y')]]
ZZ_y = ZZ_y.iloc[:,1:]
ZZ_yy = ZZ_y.reset_index()
ZZ_yy = ZZ_yy.iloc[:,1:]
print(ZZ_yy)

Z1_z = Z1.reset_index()
Z1_z = Z1_z.iloc[:,2:]
print(Z1_z)

ZZ_ss = pd.concat([ZZ_dd, ZZ_yy, Z1_z], join='inner', axis=1)
ZZ_ss = sm.add_constant(ZZ_ss)
print(ZZ_ss)
print(ZZ_ss.describe().T)


# In[10]:


# IV2SLS requires dependent variable, exogenous regressors, endogenous regressors, and instruments
Results_HT = IV2SLS(y_s, X_s, ZZ_ss).fit()
print(Results_HT.summary())


# In[ ]:





# In[13]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from statsmodels.sandbox.regression.gmm import IV2SLS

# Read CSV file
data = pd.read_csv('Data/psidextract.csv', parse_dates=['t'], index_col=['id', 't'])
print(data)

# Define variables
y  = data['lwage']
X1 = data[["occ", "south", "smsa", "ind"]]
X2 = data[["exp", "exp2", "wks", "ms", "union"]]
Z1 = data[["fem", "blk"]]
Z2 = data["ed"]
print(X1, X2, Z1, Z2)

# 함수이용
exec(open('Functions/HausmanTaylor.py').read())
HausmanTaylor(y, X1, X2, Z1, Z2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




