#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[6]:


exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[ ]:





# # PART 3: 단순회귀모형

# In[2]:


# ############################
# 단순회귀모형 전반의 이해 ###
# ############################
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import wooldridge as woo

vote1 = woo.dataWoo('vote1')

# 수식을 이용한 회귀모형 추정
x = vote1['shareA']
y = vote1['voteA']

# x,y의 상관계수
cov_xy = np.cov(x, y)[1, 0]
# x분산
var_x = np.var(x, ddof=1)
# x, y의 평균
x_bar = np.mean(x)
y_bar = np.mean(y)

# 회귀모형 추정(1): ''사용
b1 = cov_xy / var_x
b0 = y_bar - b1 * x_bar
print('\n')
print('모형의 추정결과')
print('파라미터 추정치(기울기):', b1)
print('파라미터 추정치(절편)  :', b0)
print('\n')

# 회귀모형 추정(2): f''사용
b1 = cov_xy / var_x
b0 = y_bar - b1 * x_bar
print('\n')
print(f'모형의 추정결과')
print(f'파라미터 추정치(기울기): {b1}')
print(f'파라미터 추정치(절편)  : {b0}')
print(f'\n')


# In[3]:


# statsmodels.formula library를 이용한 회귀모형 추정법
reg = smf.ols(formula='y ~ x', data=vote1)
results =reg.fit()

# 정해진 양식 전부 출력
#print(results.summary())
display(results.summary())


# In[4]:


# 정해진 양식 일부 출력
print(results.summary().tables[1])


# In[5]:


# Pandas의 Data Frame만들어서 추정결과 출력하기
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
#print(f'table: \n{table}\n')
display(table)


# In[10]:


# 원하는 지표만 선별적 출력
b = results.params
voteA_fitted = results.fittedvalues
residuals = results.resid

print('\n')
print('파라미터 추정치:', b)
print('\n')
print('회귀선 추정치:', voteA_fitted)
print('\n')
print('잔차:', residuals)


# In[11]:


# Pandas의 Data Frame만들어서 출력하기
summary_table = pd.DataFrame({'shareA': vote1['shareA'],
                      'voteA': vote1['voteA'],
                      'voteA_fitted': voteA_fitted,
                      'residuals': residuals})
#print(summary_table)
display(summary_table)


# In[12]:


# 실제 데이터 값과 회귀추정식에 의한 Fitted된 값 그래프로 그리기
plt.figure(figsize =(8, 5))
plt.scatter('shareA', 'voteA', data=vote1)
plt.plot(vote1['shareA'], results.fittedvalues, color='black', linestyle='-')
plt.ylabel('voteA')
plt.xlabel('shareA')

# 잔차의 도표 그리기 
plt.figure(figsize =(8, 2))
plt.scatter('shareA', 'residuals', data=summary_table)
plt.ylabel('Residuals')
plt.xlabel('shareA')
plt.axhline(y=0, linestyle='-')


# In[13]:


# 실제 데이터 값과 회귀추정식에 의한 Fitted된 값 그래프로 그리기
plt.figure(figsize =(8, 5))
plt.scatter('shareA', 'voteA', data=summary_table)
plt.plot('shareA', 'voteA_fitted', color='black', linestyle='-', data=summary_table)
plt.ylabel('voteA')
plt.xlabel('shareA')

# 잔차의 도표 그리기 
plt.figure(figsize =(8, 2))
plt.scatter('shareA', 'residuals', data=summary_table)
plt.ylabel('Residuals')
plt.xlabel('shareA')
plt.axhline(y=0, linestyle='-')


# In[14]:


# 추정결과로 부터 OLS기본 가정 체크하기
# 잔차항의 평균은 0? (1):
residuals_mean = np.mean(residuals)
print(residuals_mean)

# 독립변수와 잔차의 상관관계는 0? (2):
shareA_res_cov = np.cov(vote1['shareA'], residuals)[1, 0]
print(shareA_res_cov)

# 회귀선은 표본의 평균을 통과? (3):
shareA_mean = np.mean(vote1['shareA'])
voteA_fit = b[0] + b[1] * shareA_mean
print(voteA_fit)
voteA_mean = np.mean(vote1['voteA'])
print(voteA_mean)


# In[15]:


# R-square 의미이해
# R^2는 3가지 방법으로 구할 수 있음
voteA = vote1['voteA']
R2_a = np.var(results.fittedvalues, ddof=1) / np.var(voteA, ddof=1)
R2_b = 1 - np.var(residuals, ddof=1) / np.var(voteA, ddof=1)
R2_c = np.corrcoef(voteA, results.fittedvalues)[1, 0] ** 2

print(f'R2_a: {R2_a}\n')
print(f'R2_b: {R2_b}\n')
print(f'R2_c: {R2_c}\n')


# In[ ]:


# 참고: 알아 두어야 할 OLS추정법
# 1) OLS추정식을 이용한 추정법(평균, 분산,공분산 이용 )
# 2) statsmodel.formula module을 이용한 추정법
#     예)import statsmodels.api as smf 
#        smf.ols(formula='y ~ x', data=vote1)
# 3) statsmodel.api module을 이용한 추정법
#    예)  import statsmodels.api as sm
#         y = vote1['voteA']
#         X = vote1['shareA']
#         X = sm.add_constant(X)
#         sm.OLS(y, X).fit()
# 4) patsy module 사용하는 방법(다중회귀에서 설명)
# 5) 행렬연산을 이용하는 방법(다중회귀에서 설명)


# In[4]:


# ####################################
# 비선형 모형(증가율, 탄력성 추정) ###
# ####################################
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import wooldridge as woo

wage1 = woo.dataWoo('wage1')

# 선형 회귀모형
reg = smf.ols(formula='wage ~ educ', data=wage1)
results =reg.fit()

print(results.summary().tables[1])
display(results.summary().tables[1])


# In[17]:


wage1['wage_fitted'] = results.fittedvalues
wage1['residuals'] = results.resid

print(wage1.describe().T)


# In[18]:


# 실제 데이터 값과 회귀추정식에 의한 Fitted된 값 그래프로 그리기
plt.figure(figsize =(8, 5))
plt.scatter('educ', 'wage', data=wage1)
plt.plot('educ', 'wage_fitted', data=wage1, color='black', linestyle='-')
plt.ylabel('wage')
plt.xlabel('educ')

# 잔차의 도표 그리기 
plt.figure(figsize =(8, 2))
plt.scatter('educ', 'residuals', data=wage1)
plt.ylabel('Residuals')
plt.xlabel('educ')
plt.axhline(y=0, linestyle='-')


# In[5]:


# 증가율 추정 회귀모형
wage1 = woo.dataWoo('wage1')
wage1['Lwage'] = np.log(wage1['wage'])

reg = smf.ols(formula='Lwage ~ educ', data=wage1)
results =reg.fit()

print(results.summary().tables[1])
display(results.summary())


# In[21]:


wage1['Lwage_fitted'] = results.fittedvalues
wage1['residuals'] = results.resid

print(wage1.describe().T)


# In[22]:


# 실제 데이터 값과 회귀추정식에 의한 Fitted된 값 그래프로 그리기
plt.figure(figsize =(8, 5))
plt.scatter('educ', 'Lwage', data=wage1)
plt.plot('educ', 'Lwage_fitted', data=wage1, color='black', linestyle='-')
plt.ylabel('Lwage')
plt.xlabel('educ')

# 잔차의 도표 그리기 
plt.figure(figsize =(8, 2))
plt.scatter('educ', 'residuals', data=wage1)
plt.ylabel('Residuals')
plt.xlabel('educ')
plt.axhline(y=0, linestyle='-')


# In[6]:


# 탄력성 추정 회귀모형
wage1 = woo.dataWoo('wage1')

wage1['Lwage'] = np.log(wage1['wage'])
wage1['Leduc'] = np.log(wage1['educ'])

wage1.replace(-np.inf, np.nan, inplace=True)
wage1.dropna(inplace=True)

reg = smf.ols(formula='Lwage ~ Leduc', data=wage1)
results =reg.fit()

print(results.summary().tables[1])
display(results.summary())


# In[24]:


wage1['Lwage_fitted'] = results.fittedvalues
wage1['residuals'] = results.resid

print(wage1.describe().T)


# In[25]:


# 실제 데이터 값과 회귀추정식에 의한 Fitted된 값 그래프로 그리기
plt.figure(figsize =(8, 5))
plt.scatter('Leduc', 'Lwage', data=wage1)
plt.plot('Leduc', 'Lwage_fitted', data=wage1, color='black', linestyle='-')
plt.ylabel('Lwage')
plt.xlabel('Leduc')

# 잔차의 도표 그리기 
plt.figure(figsize =(8, 2))
plt.scatter('Leduc', 'residuals', data=wage1)
plt.ylabel('Residuals')
plt.xlabel('Leduc')
plt.axhline(y=0, linestyle='-')


# In[26]:


# ###########################################
# 상수항없는 모형, 기울기 없는 모형 추정) ###
# ###########################################
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import wooldridge as woo

# 한글폰트 사용하기 위한 모듈
import matplotlib as mpl
# 폰트 설정
mpl.rc('font', family='NanumGothic') 
# 유니코드에서 음수 부호 설정
mpl.rc('axes', unicode_minus=False) 

wage1 = woo.dataWoo('wage1')

# 선형 회귀모형
reg = smf.ols(formula='wage ~ educ', data=wage1)
results_1 =reg.fit()
print(results_1.summary())

# 상수항없는 회귀모형
reg = smf.ols(formula='wage ~ 0 + educ', data=wage1)
results_2 =reg.fit()
print(results_2.summary())

# 기울기 없는 회귀모형
reg = smf.ols(formula='wage ~ 1', data=wage1)
results_3 =reg.fit()
print(results_3.summary())

plt.figure(figsize =(8, 5))
# scatter plot and fitted values:
plt.scatter('educ', 'wage', data=wage1)
plt.plot(wage1['educ'], results_1.fittedvalues, color='black',
         linestyle='-', label='정상모형')
plt.plot(wage1['educ'], results_2.fittedvalues, color='black',
         linestyle=':', label='상수항 없는 모형')
plt.plot(wage1['educ'], results_3.fittedvalues, color='black',
         linestyle='-.', label='기울기 없는 모형')
plt.ylabel('wage')
plt.xlabel('educ')
plt.legend()
plt.savefig('Figures/Part 3_Simple Regression_plot2.png')


# In[4]:


#!pip show pandas
get_ipython().system('pip install --upgrade pandas')


# In[9]:


# ##########################################
# 회귀모형 몬테칼로 시뮬레이션(1 sample) ###
# ##########################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt

# 한글 폰트 사용
import matplotlib
from matplotlib import font_manager, rc
font_location='C:/Windows/Fonts/NGULIM.ttf'
font_name =font_manager.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

# set the random seed:
np.random.seed(123456)

# set sample size:
n = 1000

# set true parameters (betas and sd of u):
beta0 = 3
beta1 = 0.4

# draw a sample of size n:
x = stats.norm.rvs(4, 2, size=n)
u = stats.norm.rvs(0, 0.5, size=n)
y = beta0 + beta1 * x + u
df = pd.DataFrame({'y': y, 'x': x})

# OLS 추정치
reg = smf.ols(formula='y ~ x', data=df)
results = reg.fit()
b = results.params
print(results.summary())

# 그래프 그리기
x_range = np.linspace(-3, 11, num=140)
plt.figure(figsize =(8, 5))
plt.ylim([0, 10])
plt.scatter(x, y, s=10)
plt.plot(x_range, beta0 + beta1 * x_range, color='black',
         linestyle='-', linewidth=2, label='모회귀함수')
plt.plot(x_range, b[0] + b[1] * x_range, color='grey',
         linestyle='-', linewidth=2, label='표본회귀함수')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()


# In[10]:


# ###############################################
# 회귀모형 몬테칼로 시뮬레이션(복수의 sample) ###
# ###############################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt

# 시드값 부여
np.random.seed(123456)

# 표본수와 반복횟수
n = 1000
r = 10000

# 베타값과 그 표준편차
beta0 = 3
beta1 = 0.4

# b값 초기화
b0 = np.empty(r)
b1 = np.empty(r)

# x값은 고정(fixed)반영하기 위해 for문 바깥에 위치시킴
x = stats.norm.rvs(4, 2, size=n)

# r번 반복 실혐
for i in range(r):
    # Case 1): 오차항은 동분산, x와 상관되지 않음
    u = stats.norm.rvs(0, 0.5, size=n)
#    # Case 2): 오차항은 x와 상관
#    u_mean = np.array((x-4)/2)
#    u = stats.norm.rvs(u_mean, 0.5, size=n)
#    # Case 3): 오차항은 이분산 
#    u_var = np.array(4/np.exp(2.5)*np.exp(x))
#    u = stats.norm.rvs(0, np.sqrt(u_var), size=n)
    
    y = beta0 + beta1 * x + u
    df = pd.DataFrame({'y': y, 'x': x})

    reg = smf.ols(formula='y ~ x', data=df)
    results = reg.fit()
    b0[i] = results.params['Intercept']
    b1[i] = results.params['x']

# b1, b2추정치의 평균
b0_mean = np.mean(b0)
b1_mean = np.mean(b1)

print(f'b0_mean: {b0_mean}\n')
print(f'b1_mean: {b1_mean}\n')

# b1, b2추정치의 분산
b0_var = np.var(b0, ddof=1)
b1_var = np.var(b1, ddof=1)

print(f'b0_var: {b0_var}\n')
print(f'b1_var: {b1_var}\n')

# 그래프 그리기
plt.figure(figsize =(8, 5))
x_range = np.linspace(-3, 11, num=140)
plt.ylim([0, 10])

# 모회귀방정식 
plt.plot(x_range, beta0 + beta1 * x_range, color='black',
         linestyle='-', linewidth=2, label='모회귀함수')

# 표본회귀방정식(10000개 표본회귀 방정식중 첫번째)
plt.plot(x_range, b0[0] + b1[0] * x_range, color='grey',
         linestyle='-', linewidth=0.5, label='표본회귀함수')

# 반복회수별 표본회귀선
for i in range(1, 10):
    plt.plot(x_range, b0[i] + b1[i] * x_range, color='grey',
             linestyle='-', linewidth=0.5)
plt.ylabel('y')
plt.xlabel('x')
plt.legend()


# In[12]:


# ###############################################
# 회귀모형 몬테칼로 시뮬레이션(복수의 sample) ###
# ###############################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt

# 시드값 부여
np.random.seed(123456)

# 표본수와 반복횟수
n = 1000
r = 10000

# 베타값과 그 표준편차
beta0 = 3
beta1 = 0.4

# b값 초기화
b0 = np.empty(r)
b1 = np.empty(r)

# x값은 고정(fixed)반영하기 위해 for문 바깥에 위치시킴
x = stats.norm.rvs(4, 2, size=n)

# r번 반복 실혐
for i in range(r):
#    # Case 1): 오차항은 동분산, x와 상관되지 않음
#    u = stats.norm.rvs(0, 0.5, size=n)
    # Case 2): 오차항은 x와 상관
    u_mean = np.array((x-4)/3)
    u = stats.norm.rvs(u_mean, 0.5, size=n)
#    # Case 3): 오차항은 이분산 
#    u_var = np.array(4/np.exp(2.5)*np.exp(x))
#    u = stats.norm.rvs(0, np.sqrt(u_var), size=n)
    
    y = beta0 + beta1 * x + u
    df = pd.DataFrame({'y': y, 'x': x})

    reg = smf.ols(formula='y ~ x', data=df)
    results = reg.fit()
    b0[i] = results.params['Intercept']
    b1[i] = results.params['x']

# b1, b2추정치의 평균
b0_mean = np.mean(b0)
b1_mean = np.mean(b1)

print(f'b0_mean: {b0_mean}\n')
print(f'b1_mean: {b1_mean}\n')

# b1, b2추정치의 분산
b0_var = np.var(b0, ddof=1)
b1_var = np.var(b1, ddof=1)

print(f'b0_var: {b0_var}\n')
print(f'b1_var: {b1_var}\n')

# 그래프 그리기
plt.figure(figsize =(8, 5))
x_range = np.linspace(-3, 11, num=140)
plt.ylim([0, 10])

# 모회귀방정식 
plt.plot(x_range, beta0 + beta1 * x_range, color='black',
         linestyle='-', linewidth=2, label='모회귀함수')

# 표본회귀방정식(10000개 표본회귀 방정식중 첫번째)
plt.plot(x_range, b0[0] + b1[0] * x_range, color='grey',
         linestyle='-', linewidth=0.5, label='표본회귀함수')

# 반복회수별 표본회귀선
for i in range(1, 10):
    plt.plot(x_range, b0[i] + b1[i] * x_range, color='grey',
             linestyle='-', linewidth=0.5)
plt.ylabel('y')
plt.xlabel('x')
plt.legend()


# In[13]:


# ###############################################
# 회귀모형 몬테칼로 시뮬레이션(복수의 sample) ###
# ###############################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt

# 시드값 부여
np.random.seed(123456)

# 표본수와 반복횟수
n = 1000
r = 10000

# 베타값과 그 표준편차
beta0 = 3
beta1 = 0.4

# b값 초기화
b0 = np.empty(r)
b1 = np.empty(r)

# x값은 고정(fixed)반영하기 위해 for문 바깥에 위치시킴
x = stats.norm.rvs(4, 2, size=n)

# r번 반복 실혐
for i in range(r):
#    # Case 1): 오차항은 동분산, x와 상관되지 않음
#    u = stats.norm.rvs(0, 0.5, size=n)
#    # Case 2): 오차항은 x와 상관
#    u_mean = np.array((x-4)/3)
#    u = stats.norm.rvs(u_mean, 0.5, size=n)
    # Case 3): 오차항은 이분산 
    u_var = np.array(4/np.exp(4.5)*np.exp(x))
    u = stats.norm.rvs(0, np.sqrt(u_var), size=n)
    
    y = beta0 + beta1 * x + u
    df = pd.DataFrame({'y': y, 'x': x})

    reg = smf.ols(formula='y ~ x', data=df)
    results = reg.fit()
    b0[i] = results.params['Intercept']
    b1[i] = results.params['x']

# b1, b2추정치의 평균
b0_mean = np.mean(b0)
b1_mean = np.mean(b1)

print(f'b0_mean: {b0_mean}\n')
print(f'b1_mean: {b1_mean}\n')

# b1, b2추정치의 분산
b0_var = np.var(b0, ddof=1)
b1_var = np.var(b1, ddof=1)

print(f'b0_var: {b0_var}\n')
print(f'b1_var: {b1_var}\n')

# 그래프 그리기
plt.figure(figsize =(8, 5))
x_range = np.linspace(-3, 11, num=140)
plt.ylim([0, 10])

# 모회귀방정식 
plt.plot(x_range, beta0 + beta1 * x_range, color='black',
         linestyle='-', linewidth=2, label='모회귀함수')

# 표본회귀방정식(10000개 표본회귀 방정식중 첫번째)
plt.plot(x_range, b0[0] + b1[0] * x_range, color='grey',
         linestyle='-', linewidth=0.5, label='표본회귀함수')

# 반복회수별 표본회귀선
for i in range(1, 10):
    plt.plot(x_range, b0[i] + b1[i] * x_range, color='grey',
             linestyle='-', linewidth=0.5)
plt.ylabel('y')
plt.xlabel('x')
plt.legend()

