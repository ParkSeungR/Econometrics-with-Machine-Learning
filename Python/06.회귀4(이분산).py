#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# # 6. 이분산(Homoscedasticity)

# In[2]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[14]:


exec(open('Functions/Traditional_Econometrics_Lib.py').read())


# In[20]:


import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 데이터 읽어들이기. 필요한 변수, 관측치 선택
gpa3 = woo.dataWoo('gpa3')
gpa3 = gpa3[gpa3['spring'] == 1]
df = gpa3[['cumgpa', 'sat', 'hsperc', 'tothrs', 'female', 'black', 'white']]

# 단순기술통계량, 상관계수
print(df.describe())
print(df.corr())


# In[21]:


# 기본 회귀분석
reg = smf.ols(formula='cumgpa ~ sat + hsperc + tothrs + female + black + white', data=df)
results_default = reg.fit()
print(results_default.summary().tables[1])

# 잔차의 그래프
res = results_default.resid
yhat = results_default.fittedvalues

fig, ax = plt.subplots(2, 1, figsize=(6,4))
ax[0].set_xlabel('sat');
ax[0].set_ylabel('Residuals');
ax[0].scatter(df.sat, res, color='0.4', s=5)
ax[1].set_xlabel('yhat');
ax[1].set_ylabel('Residuals');
ax[1].scatter(yhat, res, color='0.4', s=5);
fig.subplots_adjust(hspace = 0.4);
fig.tight_layout();


# In[22]:


# 화이트(White)의 기본 강건 표준오차 추정
results_white = reg.fit(cov_type='HC0')
print(results_white.summary().tables[1])

# 화이트(White)의 수정된 표준오차(refined White SE) 추정
results_refined = reg.fit(cov_type='HC3')
print(results_refined.summary().tables[1])


# In[23]:


# 추정모형과 검증가설(hypotheses) : 흑인/백인의 차이 없음을 검증(구변수의 상관계수 -0.91)
reg = smf.ols(formula='cumgpa ~ sat + hsperc + tothrs + female + black + white',
              data=df)
hypotheses = ['black = 0', 'white = 0']

# 사용한 variance-covariance formulas에 따른 F-Tests의 차이
# 통상적인 VCOV:
results_default = reg.fit()
ftest_default = results_default.f_test(hypotheses)
fstat_default = ftest_default.statistic
fpval_default = ftest_default.pvalue
print(f'fstat_default: {fstat_default}\n')
print(f'fpval_default: {fpval_default}\n')

# 화이트(White)의 VCOV
results_hc0 = reg.fit(cov_type='HC0')
ftest_hc0 = results_hc0.f_test(hypotheses)
fstat_hc0 = ftest_hc0.statistic
fpval_hc0 = ftest_hc0.pvalue
print(f'fstat_HC0: {fstat_hc0}\n')
print(f'fpval_HC0: {fpval_hc0}\n')

# 수정된  White의 VCOV(refined White VCOV)
results_hc3 = reg.fit(cov_type='HC3')
ftest_hc3 = results_hc3.f_test(hypotheses)
fstat_hc3 = ftest_hc3.statistic
fpval_hc3 = ftest_hc3.pvalue
print(f'fstat_HC3: {fstat_hc3}\n')
print(f'fpval_HC3: {fpval_hc3}\n')


# In[25]:


# 분산함수 추정(estimation of the variance function):
df['log_e2'] = np.log(res**2)
reg_fgls = smf.ols(formula='log_e2 ~ sat + hsperc + tothrs + female + black + white',
                   data=df)
results_fgls = reg_fgls.fit()
display(results_fgls.summary())

# 가중최소자승법(WLS)
wls_weight = (1 / np.exp(results_fgls.fittedvalues))
reg_wls = smf.wls(formula='cumgpa ~ sat + hsperc + tothrs + female + black + white',
                  weights=wls_weight, data=df)
results_wls = reg_wls.fit()
display(results_wls.summary())
print(results_wls.summary())


# In[10]:


# 이분산 존재 검증을 위한 Breusch Pagan test와 White test
import wooldridge as woo
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import patsy as pt

df = woo.dataWoo('hprice1')

fig, axs = plt.subplots(2, 2, figsize=(7, 5))
sns.histplot(data=df, x="price", kde=True, ax=axs[0, 0])
sns.histplot(data=df, x="lotsize", kde=True, ax=axs[0, 1])
sns.histplot(data=df, x="sqrft", kde=True, ax=axs[1, 0])
sns.histplot(data=df, x="bdrms", kde=True, ax=axs[1, 1])
plt.tight_layout()
plt.show()


# In[11]:


# 기본모형 추정
reg = smf.ols(formula='price ~ lotsize + sqrft + bdrms', data=df)
results = reg.fit()
display(results.summary())

# F통계를 이용한 BP test(F version):
df['resid_sq'] = results.resid ** 2
reg_resid = smf.ols(formula='resid_sq ~ lotsize + sqrft + bdrms', data=df)
results_resid = reg_resid.fit()
bp_F_statistic = results_resid.fvalue
bp_F_pval = results_resid.f_pvalue
print(f'bp_F_statistic: {bp_F_statistic}\n')
print(f'bp_F_pval: {bp_F_pval}\n')

# LM통계를 이용한 BP test
y, X = pt.dmatrices('price ~ lotsize + sqrft + bdrms',
                    data=df, return_type='dataframe')
result_bp_lm = sm.stats.diagnostic.het_breuschpagan(results.resid, X)
bp_lm_statistic = result_bp_lm[0]
bp_lm_pval = result_bp_lm[1]
print(f'bp_lm_statistic: {bp_lm_statistic}\n')
print(f'bp_lm_pval: {bp_lm_pval}\n')



# In[11]:


# 이분산 존재 검증을 위한 Breusch Pagan test와 White test: log모형에 적용
import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy as pt

hprice1 = woo.dataWoo('hprice1')

# 로그 변환 모형의 정의
reg = smf.ols(formula='np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms',
              data=hprice1)
results = reg.fit()
display(results.summary())

# BP test:
y, X_bp = pt.dmatrices('np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms',
                       data=hprice1, return_type='dataframe')
result_bp = sm.stats.diagnostic.het_breuschpagan(results.resid, X_bp)
bp_statistic = result_bp[0]
bp_pval = result_bp[1]
print(f'bp_statistic: {bp_statistic}\n')
print(f'bp_pval: {bp_pval}\n')

# White test:
X_wh = pd.DataFrame({'const': 1, 'fitted_reg': results.fittedvalues,
                     'fitted_reg_sq': results.fittedvalues ** 2})
result_white = sm.stats.diagnostic.het_breuschpagan(results.resid, X_wh)
white_statistic = result_white[0]
white_pval = result_white[1]
print(f'white_statistic: {white_statistic}\n')
print(f'white_pval: {white_pval}\n')


# In[12]:


import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

k401ksubs = woo.dataWoo('401ksubs')
display(k401ksubs)
k401ksubs.describe()


# In[15]:


# 데이터 세트 정의: fsize=1만 선택
k401ksubs_sub = k401ksubs[k401ksubs['fsize'] == 1]

# 회귀모형
reg_ols = smf.ols(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                  data=k401ksubs_sub)
results_ols = reg_ols.fit(cov_type='HC2')
# print(results_ols.summary())
display(results_ols.summary())

# 가중 최소자승법(Weighted Least Squares: WLS)
wls_weight = list(1 / k401ksubs_sub['inc'])

reg_wls = smf.wls(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                  weights=wls_weight, data=k401ksubs_sub)
results_wls = reg_wls.fit()

#print(results_wls.summary())
display(results_wls.summary())

# 가중 최소자승법(화이트의 수정된SE 사용, Refined White SE)
results_white = reg_wls.fit(cov_type='HC3')
#print(results_white.summary())
display(results_white.summary())


# In[4]:


# Feasible GLS(FGLS) 추정법
import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy as pt

smoke = woo.dataWoo('smoke')
display(smoke )
smoke.describe()

# 회귀모형
formula='cigs ~ np.log(income) + np.log(cigpric) + educ + age + I(age**2) + restaurn'
reg_ols = smf.ols(formula, data=smoke)
results_ols = reg_ols.fit()
display(results_ols)

# BP test:
y, X = pt.dmatrices('cigs ~ np.log(income) + np.log(cigpric) + educ +'
                    'age + I(age**2) + restaurn',
                    data=smoke, return_type='dataframe')
result_bp = sm.stats.diagnostic.het_breuschpagan(results_ols.resid, X)
bp_statistic = result_bp[0]
bp_pval = result_bp[1]
print(f'bp_statistic: {bp_statistic}\n')
print(f'bp_pval: {bp_pval}\n')


# In[19]:


# FGLS (estimation of the variance function):
smoke['logu2'] = np.log(results_ols.resid ** 2)
reg_fgls = smf.ols(formula='logu2 ~ np.log(income) + np.log(cigpric) +'
                           'educ + age + I(age**2) + restaurn', data=smoke)
results_fgls = reg_fgls.fit()
display(results_fgls.summary())

# FGLS (WLS)
wls_weight = list(1 / np.exp(results_fgls.fittedvalues))
reg_wls = smf.wls(formula='cigs ~ np.log(income) + np.log(cigpric) +'
                          'educ + age + I(age**2) + restaurn',
                  weights=wls_weight, data=smoke)
results_wls = reg_wls.fit()
display(results_wls.summary())

