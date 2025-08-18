#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# In[2]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWDirectory\MyStock")
os.getcwd()


# In[3]:


exec(open('Functions/Traditional_Econometrics_Lib.py').read())


# A data.frame with 753 observations on 22 variables:
# * inlf: =1 if in lab frce, 1975
# * hours: hours worked, 1975
# * kidslt6: # kids < 6 years
# * kidsge6: # kids 6-18
# * age: woman’s age in yrs
# * educ: years of schooling
# * wage: est. wage from earn, hrs
# * repwage: rep. wage at interview in 1976
# * hushrs: hours worked by husband, 1975
# * husage: husband’s age
# * huseduc: husband’s years of schooling
# * huswage: husband’s hourly wage, 1975
# * faminc: family income, 1975
# * mtr: fed. marg. tax rte facing woman
# * motheduc: mother’s years of schooling
# * fatheduc: father’s years of schooling
# * unem: unem. rate in county of resid.
# * city: =1 if live in SMSA
# * exper: actual labor mkt exper
# * nwifeinc: (faminc - wage*hours)/1000
# * lwage: log(wage)
# * expersq: exper^2
# 

# In[6]:


import wooldridge as woo
import numpy as np
import pandas as pd
import linearmodels.iv as iv
import statsmodels.formula.api as smf

mroz = woo.dataWoo('mroz')

# lwage 기준 nan 제거
mroz = mroz.dropna(subset=['lwage'])

# 1) IV추정법의 이해
cov_yz = np.cov(mroz['lwage'], mroz['fatheduc'])[1, 0]
cov_xy = np.cov(mroz['educ'], mroz['lwage'])[1, 0]
cov_xz = np.cov(mroz['educ'], mroz['fatheduc'])[1, 0]
var_x = np.var(mroz['educ'], ddof=1)
x_bar = np.mean(mroz['educ'])
y_bar = np.mean(mroz['lwage'])

# OLS추정치
b_ols_man = cov_xy / var_x
print(f'b_ols_man: {b_ols_man}\n')

# IV 추정치
b_iv_man = cov_yz / cov_xz
print(f'b_iv_man: {b_iv_man}\n')

# 2) 명령어를 이용한 OLS와 IV
# 명령어를 이용한 OLS 
reg_ols = smf.ols(formula='lwage ~ educ', data=mroz)
results_ols = reg_ols.fit()
print(results_ols.summary())

# 명령어를 이용한 IV 추정
reg_iv = iv.IV2SLS.from_formula(formula='lwage ~ 1 + [educ ~ fatheduc]',
                                data=mroz)
results_iv = reg_iv.fit(cov_type='unadjusted', debiased=True)
print(results_iv)

results_iv.durbin()
results_iv.wu_hausman()



# A data.frame with 3010 observations on 34 variables:
# * id: person identifier
# * nearc2: =1 if near 2 yr college, 1966
# * nearc4: =1 if near 4 yr college, 1966
# * educ: years of schooling, 1976
# * age: in years
# * fatheduc: father’s schooling
# * motheduc: mother’s schooling
# * weight: NLS sampling weight, 1976
# * momdad14: =1 if live with mom, dad at 14
# * sinmom14: =1 if with single mom at 14
# * step14: =1 if with step parent at 14
# * reg661: =1 for region 1, 1966
# * reg662: =1 for region 2, 1966
# * reg663: =1 for region 3, 1966
# * reg664: =1 for region 4, 1966
# * reg665: =1 for region 5, 1966
# * reg666: =1 for region 6, 1966
# * reg667: =1 for region 7, 1966
# * reg668: =1 for region 8, 1966
# * reg669: =1 for region 9, 1966
# * south66: =1 if in south in 1966
# * black: =1 if black
# * smsa: =1 in in SMSA, 1976
# * south: =1 if in south, 1976
# * smsa66: =1 if in SMSA, 1966
# * wage: hourly wage in cents, 1976
# * enroll: =1 if enrolled in school, 1976
# * KWW: knowledge world of work score
# * IQ: IQ score
# * married: =1 if married, 1976
# * libcrd14: =1 if lib. card in home at 14
# * exper: age - educ - 6
# * lwage: log(wage)
# * expersq: exper^2
# 

# In[80]:


import wooldridge as woo
import numpy as np
import pandas as pd
import linearmodels.iv as iv
import statsmodels.formula.api as smf

card = woo.dataWoo('card')

# 유도형(reduced form) 모형 추정
# 교육년수
formula_educ = """educ ~ nearc4 + exper + I(exper**2) + black + smsa +
                  south + smsa66 + reg662 + reg663 + reg664 + reg665 + reg666 +
                  reg667 + reg668 + reg669"""
reg_educ = smf.ols(formula=formula_educ, data=card)
results_educ = reg_educ.fit()
display(results_educ.summary())

# 임금
formula_lwage = """lwage ~ educ + exper + I(exper**2) + black + smsa +
                   south + smsa66 + reg662 + reg663 + reg664 + reg665 +
                   reg666 + reg667 + reg668 + reg669"""
reg_lwage = smf.ols(formula=formula_lwage, data=card)
results_lwage = reg_lwage.fit()
display(results_lwage.summary())


# In[2]:


import wooldridge as woo
import numpy as np
import pandas as pd
import linearmodels.iv as iv
import statsmodels.formula.api as smf

card = woo.dataWoo('card')

# 유도형(reduced form) 모형 추정
# 교육년수
formula_educ = "educ ~ nearc4 + exper + I(exper**2) + black"
reg_educ = smf.ols(formula=formula_educ, data=card)
results_educ = reg_educ.fit()
display(results_educ.summary())

# 임금
formula_lwage = "lwage ~ educ + exper + I(exper**2) + black"
reg_lwage = smf.ols(formula=formula_lwage, data=card)
results_lwage = reg_lwage.fit()
display(results_lwage.summary())


# In[86]:


# 대변수 추정법(IV)
formula_iv = """lwage ~ 1 + exper + I(exper**2) + black + smsa + 
                south + smsa66 + reg662 + reg663 + reg664 + reg665 +
                reg666 + reg667 + reg668 + reg669 + [educ ~ nearc4]"""
reg_iv = iv.IV2SLS.from_formula(formula=formula_iv, data=card)
results_iv = reg_iv.fit(cov_type='unadjusted', debiased=True)

display(results_iv)


# In[90]:


import wooldridge as woo
import numpy as np
import pandas as pd
import linearmodels.iv as iv
import statsmodels.formula.api as smf

mroz = woo.dataWoo('mroz')

# 임금자료 존재하는 데이터세트
mroz = mroz.dropna(subset=['lwage'])

# 1단계: 교육년수 함수 추정후 적합치 구하기 
reg_educ = smf.ols(formula='educ ~ exper + I(exper**2) + motheduc + fatheduc',
                   data=mroz)
results_educ = reg_educ.fit()
mroz['educ_fitted'] = results_educ.fittedvalues
display(results_educ.summary())


# 2단계(임금함수 추정, 교육년수적합치를 독립변수로 사용
reg_lwage = smf.ols(formula='lwage ~ educ_fitted + exper + I(exper**2)',
                     data=mroz)
results_lwage = reg_lwage.fit()
display(results_lwage.summary())

# 대변수 추정(IV)
reg_iv = iv.IV2SLS.from_formula(
         formula='lwage ~ 1 + exper + I(exper**2) + [educ  ~ motheduc + fatheduc]',
         data=mroz)
results_iv = reg_iv.fit(cov_type='unadjusted', debiased=True)

display(results_iv)

# auxiliary regression:
mroz['resid_iv'] = results_iv.resids
reg_aux = smf.ols(formula='resid_iv ~ exper + I(exper**2) + motheduc + fatheduc',
                  data=mroz)
results_aux = reg_aux.fit()
display(results_aux.summary())



# A data.frame with 471 observations on 30 variables:
# * year: 1987, 1988, or 1989
# * fcode: firm code number
# * employ: # employees at plant
# * sales: annual sales, $
# * avgsal: average employee salary
# * scrap: scrap rate (per 100 items)
# * rework: rework rate (per 100 items)
# * tothrs: total hours training
# * union: =1 if unionized
# * grant: = 1 if received grant
# * d89: = 1 if year = 1989
# * d88: = 1 if year = 1988
# * totrain: total employees trained
# * hrsemp: tothrs/totrain
# * lscrap: log(scrap)
# * lemploy: log(employ)
# * lsales: log(sales)
# * lrework: log(rework)
# * lhrsemp: log(1 + hrsemp)
# * lscrap_1: lagged lscrap; missing 1987
# * grant_1: lagged grant; assumed 0 in 1987
# * clscrap: lscrap - lscrap_1; year > 1987
# * cgrant: grant - grant_1
# * clemploy: lemploy - lemploy[_n-1]
# * clsales: lavgsal - lavgsal[_n-1]
# * lavgsal: log(avgsal)
# * clavgsal: lavgsal - lavgsal[_n-1]
# * cgrant_1: cgrant[_n-1]
# * chrsemp: hrsemp - hrsemp[_n-1]
# * clhrsemp: lhrsemp - lhrsemp[_n-1]
# 

# In[112]:


import wooldridge as woo
import pandas as pd
import linearmodels.iv as iv

# 패널자료의 IV추정법
jtrain = woo.dataWoo('jtrain')
jtrain.dropna()

df = jtrain[['fcode', 'year', 'lscrap', 'hrsemp', 'grant']]
df = df.dropna(subset=['lscrap', 'hrsemp'])


# In[113]:


df = df.loc[(df['year'] == 1987) | (df['year'] == 1988), :]
df = df.set_index(['fcode', 'year'])
df.describe()


# In[115]:


# manual computation of deviations of entity means:
df['lscrap_diff1'] = \
    df.sort_values(['fcode', 'year']).groupby('fcode')['lscrap'].diff()
df['hrsemp_diff1'] = \
    df.sort_values(['fcode', 'year']).groupby('fcode')['hrsemp'].diff()
df['grant_diff1'] = \
    df.sort_values(['fcode', 'year']).groupby('fcode')['grant'].diff()
display(df)


# In[117]:


# IV regression:
df = df.dropna(subset=['lscrap_diff1'])
df


# In[119]:


reg_iv = iv.IV2SLS.from_formula(
    formula='lscrap_diff1 ~ 1 + [hrsemp_diff1 ~ grant_diff1]',
    data=df)
results_iv = reg_iv.fit(cov_type='unadjusted', debiased=True)
display(results_iv)


# ## 연립 방정식 모형

# In[11]:


import wooldridge as woo
import numpy as np
import pandas as pd
import linearmodels.iv as iv

mroz = woo.dataWoo('mroz')

# wage 기준 nan 제거
mroz = mroz.dropna(subset=['lwage'])

# 2단 최소자승법(2SLS)
reg_iv1 = iv.IV2SLS.from_formula(
    'hours ~ 1 + educ +  age + kidslt6 + nwifeinc +'
    '[np.log(wage) ~ exper + I(exper**2)]', data=mroz)
results_iv1 = reg_iv1.fit(cov_type='unadjusted', debiased=True)
display(results_iv1)

reg_iv2 = iv.IV2SLS.from_formula(
    'np.log(wage) ~ 1 + educ + exper + I(exper**2) +'
    '[hours ~ age + kidslt6 + nwifeinc]', data=mroz)
results_iv2 = reg_iv2.fit(cov_type='unadjusted', debiased=True)
display(results_iv2)

cor_u1u2 = np.corrcoef(results_iv1.resids, results_iv2.resids)[0, 1]
print(f'cor_u1u2: {cor_u1u2}\n')


# In[12]:


import wooldridge as woo
import numpy as np
import linearmodels.system as iv3

mroz = woo.dataWoo('mroz')

# restrict to non-missing wage observations:
mroz = mroz.dropna(subset=['lwage'])

# 3단 최소자승법(3SLS) 
formula = {'eq1': 'hours ~ 1 + educ + age + kidslt6 + nwifeinc +'
                  '[np.log(wage) ~ exper+I(exper**2)]',
           'eq2': 'np.log(wage) ~ 1 + educ + exper + I(exper**2) +'
                  '[hours ~ age + kidslt6 + nwifeinc]'}

reg_3sls = iv3.IV3SLS.from_formula(formula, data=mroz)

results_3sls = reg_3sls.fit(cov_type='unadjusted', debiased=True)
display(results_3sls)


# In[19]:


import wooldridge as woo
import numpy as np
import linearmodels.system as iv3

mroz = woo.dataWoo('mroz')

# restrict to non-missing wage observations:
mroz = mroz.dropna(subset=['lwage'])
hours = 'hours ~ educ + age + kidslt6 + nwifeinc + [lwage ~ exper + expersq]'
lwage = 'lwage ~ educ + exper + expersq + [hours ~ age + kidslt6 + nwifeinc]'
equations = dict(hours=hours, lwage=lwage)

# 2단 최소자승법(2SLS)
reg_iv1 = IV2SLS.from_formula(hours, data=mroz)
results_iv1 = reg_iv1.fit(cov_type='unadjusted', debiased=True)
display(results_iv1)

reg_iv2 = IV2SLS.from_formula(lwage, data=mroz)
results_iv2 = reg_iv2.fit(cov_type='unadjusted', debiased=True)
display(results_iv2)
cor_u1u2 = np.corrcoef(results_iv1.resids, results_iv2.resids)[0, 1]
print(f'cor_u1u2: {cor_u1u2}\n')

# 3단최소자승법
reg_3sls = IV3SLS.from_formula(equations, data=mroz)
results_3sls = reg_3sls.fit(cov_type='unadjusted', debiased=True)
display(results_3sls)

# System GMM Estimation
system_gmm = IVSystemGMM.from_formula(equations, data=mroz, weight_type="unadjusted")
results_gmm = system_gmm.fit(cov_type="unadjusted")
display(results_gmm)


# In[6]:


# semopy 설치 필요
get_ipython().system('pip install semopy')


# In[7]:


import pandas as pd
import wooldridge as woo
from semopy import Model

# 데이터 불러오기
mroz = woo.dataWoo('mroz')

# 변수선택 및 결측치 제거
mroz = mroz[['hours', 'lwage', 'educ', 'age', 'kidslt6', 'nwifeinc', 'exper']]
mroz['expersq'] = mroz['exper'] ** 2
mroz = mroz.dropna()

# 구조방정식 모형 정의
desc = """
# 구조방정식
hours ~ educ + age + kidslt6 + nwifeinc + lwage
lwage ~ educ + exper + expersq + hours

# 오차항 공분산 (필수)
hours ~~ hours
lwage ~~ lwage
"""

# 모형 정의 및 추정
model = Model(desc)
model.fit(mroz)  

# 추정 결과 출력
print(model.inspect())


# ## 한국의 거시모형

# In[23]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('E:/JupyterWorkingDirectory/MyStock/Data/Korea_GDP.csv',index_col='Time', parse_dates=True)

new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# In[24]:


df = data[['con', 'inv', 'gdp', 'gov', 'm2', 'rat']].loc['1995-06-30':]


# In[25]:


# 차분변수(difference variables)
df['Diff_gdp'] = df['gdp'].diff(1)
df['Diff_m2'] = df['m2'].diff(1)

# 시차변수(lag variables)
df['gdp_L1'] = df['gdp'].shift(1)
df['con_L1'] = df['con'].shift(1)
df['rat_L1'] = df['rat'].shift(1)
df = df.dropna()
display(df)


# In[27]:


# 방정식 정의
EQ_con  = 'con ~ gdp + con_L1'
EQ_inv = 'inv ~ Diff_gdp + gdp_L1 + rat_L1'
EQ_rat = 'rat ~ gdp + Diff_gdp + Diff_m2 + rat_L1'
EQ_all = dict(con=EQ_con, inv=EQ_inv, rat=EQ_rat)


# In[53]:


# 3단최소자승법
reg_3sls = IV3SLS.from_formula(EQ_all, data=df)
results_3sls = reg_3sls.fit(cov_type='unadjusted', debiased=True)
display(results_3sls)


# In[28]:


# System GMM Estimation
system_gmm = IVSystemGMM.from_formula(EQ_all, data=df, weight_type="unadjusted")
results_gmm = system_gmm.fit(cov_type="unadjusted")
display(results_gmm)


# In[30]:


df_predict = results_gmm.predict(data=df, dataframe=True)
df_predict.columns = ['con_fit', 'inv_fit', 'rat_fit']
df_all = pd.concat([df, df_predict], axis=1)
display(df_all)


# In[31]:


df_all['gdp_fit'] = df_all['con_fit']+df_all['inv_fit']+df_all['gov']
df_all.tail(50)


# In[72]:


df_all['con'].plot(legend=True,figsize=(12,6),title='Consumption')
df_all['con_fit'].plot(legend=True,figsize=(12,6))


# In[73]:


df_all['gdp'].plot(legend=True,figsize=(12,6),title='GDP')
df_all['gdp_fit'].plot(legend=True,figsize=(12,6))


# In[74]:


df_all['inv'].plot(legend=True,figsize=(12,6),title='Investment')
df_all['inv_fit'].plot(legend=True,figsize=(12,6))


# In[75]:


df_all['rat'].plot(legend=True,figsize=(12,6),title='Interest Rate')
df_all['rat_fit'].plot(legend=True,figsize=(12,6))


# In[ ]:


from linearmodels.datasets import fringe
print(fringe.DESCR)
#  Loading Dataset Information
data = fringe.load() 
data = data.dropna()
data = sm.add_constant(data)
display(data.head())

# Calculating the Coefficients using OLS
endog = ['hrearn', 'hrbens']
exog = {'hrearn': ['const', 'educ', 'exper', 'expersq', 'union', 'nrtheast', 'white'], 
        'hrbens': ['const', 'educ', 'exper', 'expersq', 'tenure', 'tenuresq', 'union', 'male']}

resids = {}
params = pd.DataFrame([]) 

for i in endog:
    results = sm.OLS(data[i], data[exog[i]]).fit() 
    resids[i] = results.resid.tolist()
    params = pd.concat([params, results.params], axis=0) 
    print('*'*80)
    print(' ')
    print(results.summary().tables[0]) 
    print(results.summary().tables[1]) 
    print(' ')

# Comparing the correlation of the errors of the equations
resids = pd.DataFrame(resids) 
display(resids.corr())

# Calculating the Coefficients using SUR

equations = {} 
for i in endog:
    equations[i] = {'dependent': data[i], 'exog': data[exog[i]]}
sys_SUR = lm.system.model.SUR(equations).fit(cov_type='unadjusted') 
print(sys_SUR.summary)

# Comparing the coefficients of the models
# The indices are equalized to concatenate the series
params.index = sys_SUR.params.index
df_comp = pd.concat([params, sys_SUR.params], axis=1)
df_comp.columns = ['0LS', 'SUR']
df_comp['Dif'] = df_comp['0LS'] - df_comp['SUR']

display(df_comp)

# Making the Breusch Pagan Test
import scipy.stats as st

# Calculating the statistic 
L = resids.corr()
L = np.tril(L, -1)**2
L = np.sum(L)
n = len(data)
M = 2
df = M*(M-1)/2 

L = n*1
p_value = 1-st.chi2.cdf(L, df)
print('Chi2_stat: ', np.round(L, 4)) 
print('p_value: ', np.round(p_value, 4))


# In[12]:


import linearmodels as lm
import statsmodels.api as sm
from linearmodels.datasets import fringe
print(fringe.DESCR)
#  Loading Dataset Information
data = fringe.load() 
data = data.dropna()
data = sm.add_constant(data)
display(data.head())

# 내생변수와 외생변수
endog = ['hrearn', 'hrbens']
exog = {'hrearn': ['const', 'educ', 'exper', 'expersq', 'union', 'nrtheast', 'white'], 
        'hrbens': ['const', 'educ', 'exper', 'expersq', 'tenure', 'tenuresq', 'union', 'male']}

# SUR 추정
equations = {} 
for i in endog:
    equations[i] = {'dependent': data[i], 'exog': data[exog[i]]}
sys_SUR = lm.system.model.SUR(equations).fit(cov_type='unadjusted') 
print(sys_SUR.summary)


# In[16]:


import linearmodels as lm
import statsmodels.api as sm
from linearmodels.datasets import fringe

#  데이터 세트 불러오기 
print(fringe.DESCR)
data = fringe.load() 
data = data.dropna()
display(data.head())

# 방정식 정의(여러가지 방법 가능)
# 참조: https://bashtage.github.io/linearmodels/system/examples/formulas.html
labeled_formula = """
{benefits: hrbens ~ educ + exper + expersq + union + south + nrtheast + nrthcen + male}
{earnings: hrearn ~ educ + exper + expersq + nrtheast + married + male}
"""
# SUR모형 추정
Model_SUR = SUR.from_formula(labeled_formula, data)
Result_SUR = labels_mod.fit(cov_type="unadjusted")
print(Result_SUR)


# In[20]:


import linearmodels as lm
import statsmodels.api as sm
from linearmodels import IV2SLS, IV3SLS, SUR, IVSystemGMM
from linearmodels.datasets import mroz

# 데이터 읽어오기
data = mroz.load()
data = data[["hours", "educ", "age", "kidslt6", "nwifeinc", "lwage", "exper", "expersq"]]
data = data.dropna()

# 2단 최소자승법
# 근로시간
hours = "hours ~ educ + age + kidslt6 + nwifeinc + [lwage ~ exper + expersq]"
hours_mod = IV2SLS.from_formula(hours, data)
hours_res = hours_mod.fit(cov_type="unadjusted")
print(hours_res)

# 임금
lwage = "lwage ~ educ + exper + expersq + [hours ~ age + kidslt6 + nwifeinc]"
lwage_mod = IV2SLS.from_formula(lwage, data)
lwage_res = lwage_mod.fit(cov_type="unadjusted")
print(lwage_res)

# 3단 최소자승법
equations = dict(hours=hours, lwage=lwage)
system_2sls = IV3SLS.from_formula(equations, data)
system_2sls_res = system_2sls.fit(method="ols", cov_type="unadjusted")
print(system_2sls_res)

# System GMM Estimation
equations = dict(
    hours="hours ~ educ + age + kidslt6 + nwifeinc + [lwage ~ exper + expersq]",
    lwage="lwage ~ educ + exper + expersq + [hours ~ age + kidslt6 + nwifeinc]",
                 )
system_gmm = IVSystemGMM.from_formula(equations, data, weight_type="unadjusted")
system_gmm_res = system_gmm.fit(cov_type="unadjusted")
print(system_gmm_res)

system_gmm = IVSystemGMM.from_formula(equations, data, weight_type="robust")
system_gmm_res = system_gmm.fit(cov_type="robust", iter_limit=100)
print("Number of iterations: " + str(system_gmm_res.iterations))
print(system_gmm_res)


# In[ ]:





# # 구조방정식 모형

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWDirectory\MyStock")
os.getcwd()


# In[13]:


exec(open('E:/JupyterWDirectory/MyStock/Functions/My_sem_analysis.py', encoding='utf-8').read())


# In[73]:


#!pip install factor-analyzer


# In[14]:


# 데이터 로드
cols = ['attend', 'confed', 'conlabor', 'degree', 'educ', 'god', 'madeg', 'maeduc', 'padeg', 'paeduc', 'polviews', 'pray', 'relpersn', 'speduc']
data = pd.read_excel("G:/1. Stata 저술 본문/statpgm/part7/VII-6-2-SEMGSS.xlsx", usecols=cols)
data.head()


# In[15]:


# 모델 명세 정의
model_spec = {
    'measurement_model': {
        'Peduc': ['paeduc', 'padeg', 'maeduc', 'madeg'],
        'Reduc': ['educ', 'speduc', 'degree'],
        'Liberalism': ['relpersn', 'attend', 'god', 'pray'],
        'Religiousity': ['polviews', 'conlabor', 'confed']
    },
    'structural_model': [
        ('Peduc', 'Reduc'),
        ('Peduc', 'Religiousity'),
        ('Religiousity', 'Liberalism')
    ],
    'covariances': [
        ('maeduc', 'padeg'),
        ('conlabor', 'confed')
    ]
}

# SEM 분석 실행
results = my_sem_analysis(data, model_spec)


# In[3]:


get_ipython().system('pip install pingouin')


# In[47]:


import pandas as pd
import pingouin as pg
from sklearn.decomposition import PCA
from statsmodels.multivariate.factor import Factor
from semopy import Model
import seaborn as sns
import matplotlib.pyplot as plt

def sem_auto_report(df, sem_model_desc):
    # pandas 출력 옵션 설정 (함수 시작 시)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    results = {}

    # 1. 기술통계
    desc = df.describe()
    print("\n[기술통계]")
    print(desc)
    results['Descriptive'] = desc

    # 2. 상관행렬
    corr = pg.pairwise_corr(df, method='pearson')
    print("\n[상관행렬 (p-value 포함)]")
    print(corr[['X', 'Y', 'r', 'p-unc']])
    results['Correlation'] = corr

    # 3. PCA (설명된 분산 비율)
    pca = PCA()
    pca.fit(df)
    print("\n[PCA - 설명된 분산 비율]")
    print(pca.explained_variance_ratio_)
    results['PCA Explained Variance Ratio'] = pca.explained_variance_ratio_

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
    plt.title('PCA Scree Plot')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    # 4. Cronbach Alpha (SEM 정의 스크립트에서 직접 파싱)
    print("\n[Cronbach Alpha - SEM 정의 스크립트 이용]")
    measurement_lines = [line.strip() for line in sem_model_desc.strip().split('\n') if '=~' in line]

    for line in measurement_lines:
        lhs, rhs = line.split('=~')
        latent_var = lhs.strip()
        observed_vars = [var.strip() for var in rhs.strip().split('+')]
        alpha, ci = pg.cronbach_alpha(df[observed_vars])
        print(f"[{latent_var}] Cronbach's Alpha: {alpha:.3f}, 95% CI: {ci}")
        results[f'Cronbach Alpha {latent_var}'] = {'Alpha': alpha, 'CI': ci, 'Observed Variables': observed_vars}

    # 5. SEM 추정 및 결과 확인
    model = Model(sem_model_desc)
    model.fit(df)

    print("\n[SEM 추정된 파라미터 (Standardized Estimates)]")
    est_params = model.inspect()
    print(est_params)
    results['SEM Parameters'] = est_params

    print("\n[SEM 적합도 지표]")
    modelfit_stats = model.inspect('modelfit')
    print(modelfit_stats)
    results['SEM Fit Statistics'] = modelfit_stats

    print("\n[SEM 직접, 간접, 총효과]")
    effects = model.inspect('effects')
    print(effects)
    results['SEM Effects'] = effects

    # 6. FA (잠재변수 수 기준)
    latent_vars = model.vars['latent']
    fa = Factor(df, n_factor=len(latent_vars))
    fa_res = fa.fit()

    print("\n[FA Loadings]")
    print(fa_res.loadings)
    results['FA Loadings'] = fa_res.loadings

    # FA - 강제 PCA로 Eigen Values 추가 출력
    print("\n[FA (PCA 기반 Eigen Values)]")
    pca = PCA()
    pca.fit(df)
    print("Eigen Values:", pca.explained_variance_)
    results['FA Eigen Values'] = pca.explained_variance_

    return results


# In[28]:


# 데이터 로드
# 불러올 변수(컬럼) 리스트
cols = ['attend', 'confed', 'conlabor', 'degree', 'educ', 'god', 'madeg', 'maeduc', 'padeg', 'paeduc', 'polviews', 'pray', 'relpersn', 'speduc']

# 특정 변수만 불러오기
data = pd.read_excel("G:/1. Stata 저술 본문/statpgm/part7/VII-6-2-SEMGSS.xlsx", usecols=cols)

print(data.head())


# In[29]:


sem_model_desc = """
Peduc =~ paeduc + padeg + maeduc + madeg
Reduc =~ educ + speduc + degree
Liberalism =~ relpersn + attend + god + pray
Religiousity =~ polviews + conlabor + confed
Reduc ~ Peduc
Religiousity ~ Peduc
Liberalism ~ Religiousity
maeduc ~~ padeg
conlabor ~~ confed
"""


# In[21]:


get_ipython().system('pip install graphviz')


# In[48]:


result = sem_auto_report(data, sem_model_desc)


# In[53]:


# 5. SEM 추정 및 결과 확인
model = Model(sem_model_desc)
res = model.fit(data)
print(res)


# In[54]:


print("\n[SEM 추정된 파라미터 (Standardized Estimates)]")
est_params = model.inspect()
print(est_params)


# In[2]:


#stats =  model.calc_stats()
stats =  semopy.calc_stats(model)
basic_stats = stats[['DoF','chi2', 'RMSEA', 'NFI', 'TLI', 'CFI', 'GFI']]
print(basic_stats)


# In[ ]:


print("\n[SEM 직접, 간접, 총효과]")
effects = model.inspect('effects')
print(effects)
results['SEM Effects'] = effects


# In[ ]:





# In[ ]:





# In[ ]:


import semopy as sem
import pandas as pd
import numpy as np
from statsmodels.stats.moment_helpers import corr2cov
# Data Import
data = pd.read_csv(‘houghton.csv')
# Transforming correlations to Variance Covariance Matrix
variables = ["work1","work2","work3","happy","mood1","mood2","perform1","perform2","approval","beliefs","selftalk","imagery"]
data2= data[variables]
covMatrix = corr2cov(data2[1:13], data2[13:14])
X = np.tril(np.array(covMatrix))
X_low = np.tril(X)
X_low = X_low.T
for ind, i in enumerate(X_low):
    for ind2, j in enumerate(i):
        if(ind == ind2):
            X_low[ind][ind2]=0
covFull = X+X_low

# Creating Dataframe for SEM analysis in python
dataframe = pd.DataFrame(data=covFull, index=variables, columns=variables)

# Describing model for SEM
mod = """
# measurement model
Construc =~ beliefs + selftalk + imagery
Dysfunc =~ perform1 + perform2 + approval
WellBe =~ happy + mood1 + mood2
JobSat =~ work1 + work2 + work3
# error covariance
happy ~~ mood2 
# structural part
Dysfunc ~ Construc
WellBe ~ Construc + Dysfunc
JobSat ~ Construc + Dysfunc + WellBe
model = sem.Model(mod)
res = model.fit(data=None, obj="MLW",solver='SLSQP',cov=dataframe, n_samples=263)

estimates = model.inspect(std_est=True,)
stats =  sem.calc_stats(model)
basic_stats = stats[['DoF','chi2', 'RMSEA', 'NFI', 'TLI', 'CFI', 'GFI']]
sem.semplot(model, "model_houghton.png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




