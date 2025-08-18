#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# # 8. 모형설정(Model Specification) 및 데이터관련 문제

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[2]:


exec(open('Functions/Traditional_Econometrics_Lib.py').read())


# In[2]:


get_ipython().system('pip install --upgrade pandas')


# In[6]:


# 모형설정오류 테스트(RESET)
import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as smo

hprice1 = woo.dataWoo('hprice1')
display(hprice1)
hprice1.describe()


# In[7]:


# 최소자승법 적용
reg = smf.ols(formula='price ~ lotsize + sqrft + bdrms', data=hprice1)
results = reg.fit()
display(results.summary())

# RESET test를 위한 회귀분석
hprice1['fitted_sq'] = results.fittedvalues ** 2
hprice1['fitted_cub'] = results.fittedvalues ** 3
reg_reset = smf.ols(formula='price ~ lotsize + sqrft + bdrms +'
                            'fitted_sq + fitted_cub', data=hprice1)
results_reset = reg_reset.fit()
display(results_reset.summary())

# RESET test (H0: 적합된 값을 포함한 모든 변수의 파라미터=0)
hypotheses = ['fitted_sq = 0', 'fitted_cub = 0']
ftest_man = results_reset.f_test(hypotheses)
fstat_man = ftest_man.statistic
fpval_man = ftest_man.pvalue

print(f'fstat_man: {fstat_man}\n')
print(f'fpval_man: {fpval_man}\n')

# 함수를 이용한 RESET 테스트: .reset_ramsey이용
reset_output = smo.reset_ramsey(res=results, degree=3)
fstat_auto = reset_output.statistic
fpval_auto = reset_output.pvalue

print(f'fstat_auto: {fstat_auto}\n')
print(f'fpval_auto: {fpval_auto}\n')


# In[4]:


# Non-Nested Test
import wooldridge as woo
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import stargazer.stargazer as sg

hprice1 = woo.dataWoo('hprice1')

# 2개의 대안모형(two alternative models)
reg1 = smf.ols(formula='price ~        lotsize +         sqrft  + bdrms', data=hprice1)
results1 = reg1.fit()
#display(results1.summary())

reg2 = smf.ols(formula='price ~ np.log(lotsize) + np.log(sqrft) + bdrms', data=hprice1)
results2 = reg2.fit()
#display(results2.summary())

# Davidson & MacKinnon test
# 포괄적 모형(comprehensive model)
reg3 = smf.ols(formula='price ~ lotsize + sqrft + bdrms + np.log(lotsize) + np.log(sqrft)', data=hprice1)
results3 = reg3.fit()
#display(results3.summary())

# stargazer를 이용하여 추정결과 정리
stargazer = sg.Stargazer([results1, results2, results3])
#display(stargazer)
print(stargazer)

# model 1과 포괄적 모형 
anovaResults1 = sm.stats.anova_lm(results1, results3)
print(f'anovaResults1: \n{anovaResults1}\n')

# model 2와 포괄적 모형 
anovaResults2 = sm.stats.anova_lm(results2, results3)
print(f'anovaResults2: \n{anovaResults2}\n')


# In[9]:


# ###############################
# 측정오차(Measurement Error) ###
# ###############################

# 종속변수 y에 측정오차의 문제가 있을 때
import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns

# random seed
np.random.seed(123456)

# sample size와 시뮬레이션 반복횟수
n = 1000
r = 10000

# 모수값 (parameters: betas)
beta0 = 2
beta1 = 0.4

# 측정오차가 없을때와 있을때의 기울치 추정치 배열
b1    = np.empty(r)
b1_me = np.empty(r)

# 시뮬레이션 반복시에 고정된 X값(condinal regression)
x = stats.norm.rvs(4, 2, size=n)

# 시뮬레이션 r회 반복
for i in range(r):
    # 오차항 u
    u = stats.norm.rvs(0, 1, size=n)

    # 종속변수 생성(ystar)
    ystar = beta0 + beta1 * x + u
    # 측정오차가 있는 y값 생성
    e0 = stats.norm.rvs(0, 1, size=n)
    y = ystar + e0
    df = pd.DataFrame({'ystar': ystar, 'y': y, 'x': x})

    # 측정오차가 없는 ystar에 대한 회귀분석, 기울기 추정치 보관
    reg_star = smf.ols(formula='ystar ~ x', data=df)
    results_star = reg_star.fit()
    b1[i] = results_star.params['x']

    # 측정오차가 있는 y에 대한 회귀분석, 기울기 추정치 보관
    reg_me = smf.ols(formula='y ~ x', data=df)
    results_me = reg_me.fit()
    b1_me[i] = results_me.params['x']

# 종속변수에 측정오차가 있을 때와 없을때 기울기 파라미터 추정치 평균
b1_mean = np.mean(b1)
b1_me_mean = np.mean(b1_me)
print(f'b1_mean: {b1_mean}\n')
print(f'b1_me_mean: {b1_me_mean}\n')

# 종속변수에 측정오차가 있을 때와 없을때 기울기 파라미터 추정치의 분산
b1_var = np.var(b1, ddof=1)
b1_me_var = np.var(b1_me, ddof=1)
print(f'b1_var: {b1_var}\n')
print(f'b1_me_var: {b1_me_var}\n')


# In[4]:


sns.displot(b1_me, kde=True, bins=30)


# In[5]:


# 독립변수 X에 측정오차의 문제가 있을 때
import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.formula.api as smf

# random seed 설정
np.random.seed(123456)

# sample size와 시뮬레이션 반복횟수
n = 1000
r = 10000

# 모수값 (parameters: betas)
beta0 = 2
beta1 = 0.4

# 측정오차가 없을때와 있을때의 기울치 추정치 배열
b1 = np.empty(r)
b1_me = np.empty(r)

# 시뮬레이션 반복시에 고정된 X값(conditional regression)
xstar = stats.norm.rvs(4, 2, size=n)

# 시뮬레이션 r회 반복
for i in range(r):
    # draw a sample of u:
    u = stats.norm.rvs(0, 1, size=n)
    # 측정오차없는 종속변수 생성(y)
    y = beta0 + beta1 * xstar + u
    # 측정오차가 있는 x값 생성
    e1 = stats.norm.rvs(0, 1, size=n)
    x = xstar + e1
    df = pd.DataFrame({'y': y, 'xstar': xstar, 'x': x})
    # 측정오차가 있는 xstary에 대한 회귀분석, 기울기 추정치 보관
    reg_star = smf.ols(formula='y ~ xstar', data=df)
    results_star = reg_star.fit()
    b1[i] = results_star.params['xstar']
    # 측정오차가 없는 x에 대한 회귀분석, 기울기 추정치 보관
    reg_me = smf.ols(formula='y ~ x', data=df)
    results_me = reg_me.fit()
    b1_me[i] = results_me.params['x']

# 독립변수에 측정오차가 있을 때와 없을때 기울기 파라미터 추정치 평균
b1_mean = np.mean(b1)
b1_me_mean = np.mean(b1_me)
print(f'b1_mean: {b1_mean}\n')
print(f'b1_me_mean: {b1_me_mean}\n')

# 독립변수에 측정오차가 있을 때와 없을때 기울기 파라미터 추정치의 분산
b1_var = np.var(b1, ddof=1)
b1_me_var = np.var(b1_me, ddof=1)
print(f'b1_var: {b1_var}\n')
print(f'b1_me_var: {b1_me_var}\n')


# In[8]:


sns.displot(b1_me, kde=True, bins=30)


# In[4]:


# 누락치의 문제

import numpy as np
import pandas as pd
import scipy.stats as stats

# 누락치, 무한대값을 가진 가상적 numpy array만들기
x = np.array([-1, 0, 1, 2, np.nan, -np.inf, np.inf])

# 로그변환: -1은 정의안됨, 0은 -inf, +inf는 +inf
logx = np.log(x)

# 역수: 1/0=inf, 1/inf=0
invx = np.array(1 / x)

# 정규분포 cdf
ncdf = np.array(stats.norm.cdf(x))

# 누락치 여부: nan=True
isnanx = np.isnan(x)

# 이상에서 계산된 값을 DataFrame으로 만들기
results = pd.DataFrame({'x': x, 'log(x)': logx, 'inv(x)': invx,
                        'ncdf': ncdf, 'isnanx': isnanx})
display(results)

# -inf, inf값을 누락치로 만들기
results = results.replace([np.inf, -np.inf], np.nan)
display(results)

# 누락치 행(row) 삭제하기
results = results.dropna()

# 결과 확인
display(results)




# In[16]:


# 누락치 확인 
import wooldridge as woo
import pandas as pd

lawsch85 = woo.dataWoo('lawsch85')
#display(lawsch85)

# 단순통계량 count에서도 누락치 확인 가능
display(lawsch85.describe().T)

# 데이터 프레임에 있는 모든 변수들의 누락치 계산 
miss_all = lawsch85.isna()
miss_num = miss_all.sum(axis=0)
display(miss_num)


# In[22]:


# 이상치(outlier) 확인
import wooldridge as woo
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

rdchem = woo.dataWoo('rdchem')
display(rdchem)
display(rdchem.describe().T)

# OLS regression:
reg = smf.ols(formula='rdintens ~ sales + profmarg', data=rdchem)
results = reg.fit()

# studentized residuals 계산
studres = results.get_influence().resid_studentized_external

# display extreme values:
studres_max = np.max(studres)
studres_min = np.min(studres)
print(f'studres_max: {studres_max}\n')
print(f'studres_min: {studres_min}\n')

sns.displot(studres, kde=True, bins=10)


# In[23]:


# Least Absolute Deviations(LAD) 추정법

import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

rdchem = woo.dataWoo('rdchem')

# 최소자승법
reg_ols = smf.ols(formula='rdintens ~ I(sales/1000) + profmarg', data=rdchem)
results_ols = reg_ols.fit()
display(results_ols.summary())

# LAD 회귀: 
reg_lad = smf.quantreg(formula='rdintens ~ I(sales/1000) + profmarg', data=rdchem)
results_lad = reg_lad.fit(q=.5)
display(results_lad.summary())


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Generate sample data
np.random.seed(0)
x =np.linspace(-5,5,num=50)
y =2.0 +1.5 *x +3.0 *x**2 +np.random.normal(scale=3.0,size=x.shape)

# Define the nonlinear function
def quadratic_func(x,a,b,c):
    return a +b *x +c *x**2
# Fit the nonlinear model
popt,pcov =curve_fit(quadratic_func,x,y)

# Visualize the results
plt.scatter(x,y,label='Data')
plt.plot(x,quadratic_func(x,*popt),'r-',label='Fit')
plt.legend()
plt.show()


# In[25]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize

# 데이터 불러오기
data = pd.read_csv('data/KorAutoCD.csv')
data.set_index('year', inplace=True)
display(data) 

# 로그변환
data['lnY'] = np.log(data['y'])
data['lnL'] = np.log(data['l'])
data['lnK'] = np.log(data['k'])

# Translog function 추정을 위한 변수생성
data['lnK_squared'] = 0.5*data['lnK'] ** 2
data['lnL_squared'] = 0.5*data['lnL'] ** 2
data['lnK_lnL'] = data['lnK'] * data['lnL']
display(data)

# Cobb-Douglas 생산함수 추정
reg_CD = smf.ols(formula='lnY ~ lnL + lnK', data=data)
results_CD = reg_CD.fit()
display(results_CD.summary())

# Translog 생산함수 추정
reg_TL = smf.ols(formula='lnY ~ lnL + lnK + lnK_squared+ lnL_squared +lnK_lnL', data=data)
results_TL = reg_TL.fit()
display(results_TL.summary())

# CES 생산함수 추정
def ces_production(params, y, l, k):
    A, delta, rho = params
    return A * (delta * l*rho + (1 - delta) * k**rho)**(1/rho)

# 최소화를 위한 목적함수
def objective(params, y, l, k):
    estimated_y = ces_production(params, y, l, k)
    return np.sum((y - estimated_y)**2)

# 초깃값
initial_params = [1, 0.5, 0.5]

# 추정
result = minimize(objective, initial_params, args=(data['y'], data['l'], data['k']), method='Nelder-Mead')

# 추정된 파라미터
estimated_params = result.x
display(estimated_params)

# 파라미터 정리
A_hat, delta_hat, rho_hat = estimated_params
sigma_hat = 1 / (1 - rho_hat)


print(f"Estimated A: {A_hat}")
print(f"Estimated delta: {delta_hat}")
print(f"Estimated rho: {rho_hat}")
print(f"Estimated sigma (elasticity of substitution): {sigma_hat}")


# In[23]:





# In[24]:





# In[18]:


# CES 생산함수 추정
def ces_production(params, y, l, k):
    A, delta, rho = params
    return A * (delta * l*rho + (1 - delta) * k**rho)**(1/rho)

# 최소화를 위한 목적함수
def objective(params, y, l, k):
    estimated_y = ces_production(params, y, l, k)
    return np.sum((y - estimated_y)**2)

# 초깃값
initial_params = [1, 0.5, 0.5]

# 추정
result = minimize(objective, initial_params, args=(data['y'], data['l'], data['k']), method='Nelder-Mead')

# 추정된 파라미터
estimated_params = result.x
display(estimated_params)

# 파라미터 정리
A_hat, delta_hat, rho_hat = estimated_params
sigma_hat = 1 / (1 - rho_hat)


print(f"Estimated A: {A_hat}")
print(f"Estimated delta: {delta_hat}")
print(f"Estimated rho: {rho_hat}")
print(f"Estimated sigma (elasticity of substitution): {sigma_hat}")


# In[ ]:


# 구조변화 검정
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid

# Load data
data = pd.read_csv('your_timeseries_data.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Perform Chow test
breakpoint = 50
X = sm.add_constant(data.index.values)
X1, y1 = X[:breakpoint], data['value'][:breakpoint]
X2, y2 = X[breakpoint:], data['value'][breakpoint:]
model1 = sm.OLS(y1, X1).fit()
model2 = sm.OLS(y2, X2).fit()
chow_test = sm.stats.chowtest(model1, model2, breakpoint)
print(f'Chow Test F-statistic: {chow_test[0]}, p-value: {chow_test[1]}')

# Perform CUSUM test
model = sm.OLS(data['value'], sm.add_constant(data.index.values)).fit()
cusum_test = breaks_cusumolsresid(model.resid)
print(f'CUSUM Test: {cusum_test}')

# Fit Markov Switching Model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
model = MarkovRegression(data['value'], k_regimes=2, trend='c', switching_variance=True)
result = model.fit()
print(result.summary())

