#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# In[1]:


import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[2]:


exec(open('Functions/Traditional_Econometrics_Lib.py').read())


# Usage
# data('prminwge')
# Format
# A data.frame with 38 observations on 25 variables:
# * year: 1950-1987
# * avgmin: weighted avg min wge, 44 indust
# * avgwage: wghted avg hrly wge, 44 indust
# * kaitz: Kaitz min wage index
# * avgcov: wghted avg coverage, 8 indust
# * covt: economy-wide coverage of min wg
# * mfgwage: avg manuf. wage
# * prdef: Puerto Rican price deflator
# * prepop: PR employ/popul ratio
# * prepopf: PR employ/popul ratio, alter.
# * prgnp: PR GNP
# * prunemp: PR unemployment rate
# * usgnp: US GNP
# * t: time trend: 1 to 38
# * post74: time trend: starts in 1974
# * lprunemp: log(prunemp)
# * lprgnp: log(prgnp)
# * lusgnp: log(usgnp)
# * lkaitz: log(kaitz)
# * lprun_1: lprunemp[_n-1]
# * lprepop: log(prepop)
# * lprep_1: lprepop[_n-1]
# * mincov: (avgmin/avgwage)*avgcov
# * lmincov: log(mincov)
# * lavgmin: log(avgmin)
# 
# 

# In[4]:


import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

prminwge = woo.dataWoo('prminwge')
T = len(prminwge)
prminwge['time'] = prminwge['year'] - 1949
prminwge.index = pd.date_range(start='1950', periods=T, freq='Y').year

# OLS
reg = smf.ols(formula='np.log(prepop) ~ np.log(mincov) + np.log(prgnp) +'
                      'np.log(usgnp) + time', data=prminwge)
# SE:
results_regu = reg.fit()
display(results_regu.summary())

# HAC SE
results_hac = reg.fit(cov_type='HAC', cov_kwds={'maxlags': 2})
display(results_hac.summary())


# In[8]:


import wooldridge as woo
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy as pt

phillips = woo.dataWoo('phillips')
T = len(phillips)
display(phillips)


# In[9]:


# 연도변수 인덱스
df = phillips[['year', 'unem', 'inf']]
date_range = pd.date_range(start='1948', periods=T, freq='Y')
df.index = date_range.year

# 차분변수 생성
df['inflation'] =  df['inf']
df['inflation_d'] = df['inflation'].diff()

# 1) Expectations-augmented Phillips curve 추정
reg_ea = smf.ols(formula='inflation_d ~ unem', data=df)
results_ea = reg_ea.fit()
display(results_ea.summary())

# 잔차의 AR(1)함수 추정
df['resid_ea'] = results_ea.resid
df['resid_ea_lag1'] = df['resid_ea'].shift(1)
reg = smf.ols(formula='resid_ea ~ resid_ea_lag1', data=df)
results = reg.fit()
display(results.summary())

# 2) Static Phillips curve 추정
reg_s = smf.ols(formula= 'inflation ~ unem', data=df)
results_s = reg_s.fit()
display(results_s.summary())

# 잔차 AR(1) 
df['resid_s'] = results_s.resid
df['resid_s_lag1'] = df['resid_s'].shift(1)
reg = smf.ols(formula='resid_s ~ resid_s_lag1', data=df)
results = reg.fit()
display(results.summary())

# 더빈-왓슨 검정(DW tests)
DW_ea = sm.stats.stattools.durbin_watson(results_ea.resid)
DW_s = sm.stats.stattools.durbin_watson(results_s.resid)
print(f'DW_ea: {DW_ea}\n')
print(f'DW_s: {DW_s}\n')

# 브로슈-갓프리 검정(Breusch Godfrey test)
bg_result_ea = sm.stats.diagnostic.acorr_breusch_godfrey(results_ea, nlags=3)
fstat_ea = bg_result_ea[2]
fpval_ea = bg_result_ea[3]
print(f'fstat_ea: {fstat_ea}\n')
print(f'fpval_ea: {fpval_ea}\n')

bg_result_s = sm.stats.diagnostic.acorr_breusch_godfrey(results_s, nlags=3)
fstat_s = bg_result_s[2]
fpval_s = bg_result_s[3]
print(f'fstat_s: {fstat_s}\n')
print(f'fpval_s: {fpval_s}\n')


# In[14]:


# 코크레인-오컷 추정법Cochrane-Orcutt estimation)
# 1) Expectations-augmented Phillips curve의 CORC 추정
y, X = pt.dmatrices('inflation_d ~ unem', data=df, return_type='dataframe')
reg = sm.GLSAR(y, X)
CORC_results_ea = reg.iterative_fit(maxiter=100)
display(CORC_results_ea.summary())
print(CORC_results_ea.summary())

# 2) Static Phillips curve의 CORC 추정
y, X = pt.dmatrices('inflation ~ unem', data=df, return_type='dataframe')
reg = sm.GLSAR(y, X)
CORC_results_s = reg.iterative_fit(maxiter=100)
display(CORC_results_s.summary())
print(CORC_results_s.summary())


# In[17]:


# 뉴이-웨스트 표준오차(Newey-West Standard error)
# 1) Expectations-augmented Phillips curve의 CORC 추정
y, X = pt.dmatrices('inflation_d ~ unem', data=df, return_type='dataframe')
reg_ea = sm.GLSAR(y, X)
# HAC SE
results_hac_ea = reg_ea.fit(cov_type='HAC', cov_kwds={'maxlags': 2})
print(results_hac_ea.summary())

# 2) Static Phillips curve의 CORC 추정
y, X = pt.dmatrices('inflation ~ unem', data=df, return_type='dataframe')
reg_s = sm.GLSAR(y, X)
# HAC SE
results_hac_s = reg_s.fit(cov_type='HAC', cov_kwds={'maxlags': 2})
print(results_hac_s.summary())


# A data.frame with 131 observations on 31 variables:
# * chnimp: Chinese imports, bar. chl.
# * bchlimp: total imports bar. chl.
# * befile6: =1 for all 6 mos before filing
# * affile6: =1 for all 6 mos after filing
# * afdec6: =1 for all 6 mos after decision
# * befile12: =1 all 12 mos before filing
# * affile12: =1 all 12 mos after filing
# * afdec12: =1 all 12 mos after decision
# * chempi: chemical production index
# * gas: gasoline production
# * rtwex: exchange rate index
# * spr: =1 for spring months
# * sum: =1 for summer months
# * fall: =1 for fall months
# * lchnimp: log(chnimp)
# * lgas: log(gas)
# * lrtwex: log(rtwex)
# * lchempi: log(chempi)
# * t: time trend
# * feb: =1 if month is feb
# * mar: =1 if month is march
# * apr:
# * may:
# 

# In[11]:


import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

barium = woo.dataWoo('barium')
T = len(barium)

# 시간 인덱스
barium.index = pd.date_range(start='1978-02', periods=T, freq='M')

# 함수추정
reg = smf.ols(formula='np.log(chnimp) ~ np.log(chempi) + np.log(gas) +'
                      'np.log(rtwex) + befile6 + affile6 + afdec6',
              data=barium)
results = reg.fit()
display(results.summary())

# 수식을 이용한 Breusch Godfrey test
barium['resid'] = results.resid
barium['resid_lag1'] = barium['resid'].shift(1)
barium['resid_lag2'] = barium['resid'].shift(2)
barium['resid_lag3'] = barium['resid'].shift(3)

reg_manual = smf.ols(formula='resid ~ resid_lag1 + resid_lag2 + resid_lag3 +'
                             'np.log(chempi) + np.log(gas) + np.log(rtwex) +'
                             'befile6 + affile6 + afdec6', data=barium)
results_manual = reg_manual.fit()
display(results_manual.summary())

hypotheses = ['resid_lag1 = 0', 'resid_lag2 = 0', 'resid_lag3 = 0']
ftest_manual = results_manual.f_test(hypotheses)
fstat_manual = ftest_manual.statistic
fpval_manual = ftest_manual.pvalue
print(f'fstat_manual: {fstat_manual}\n')
print(f'fpval_manual: {fpval_manual}\n')

# 함수를 이용한 Breusch Godfrey test:
bg_result = sm.stats.diagnostic.acorr_breusch_godfrey(results, nlags=3)
fstat_auto = bg_result[2]
fpval_auto = bg_result[3]
print(f'fstat_auto: {fstat_auto}\n')
print(f'fpval_auto: {fpval_auto}\n')

# 더빈-왓슨 검정(DW tests)
DW = sm.stats.stattools.durbin_watson(results.resid)
print(f'DW: {DW}\n')

# Cochrane-Orcutt 방법에 의한 모형추정 
y, X = pt.dmatrices('np.log(chnimp) ~ np.log(chempi) + np.log(gas) +'
                    'np.log(rtwex) + befile6 + affile6 + afdec6',
                    data=barium, return_type='dataframe')
reg = sm.GLSAR(y, X)
CORC_results = reg.iterative_fit(maxiter=100)
display(CORC_results.summary())


# A data.frame with 691 observations on 8 variables:
# * price: NYSE stock price index
# * return: 100*(p - p(-1))/p(-1))
# * return_1: lagged return
# okun 115
# * t:
# * price_1:
# * price_2:
# * cprice: price - price_1
# * cprice_1: lagged cprice
# 

# In[4]:


import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

nyse = woo.dataWoo('nyse')
nyse['ret'] = nyse['return']
nyse['ret_lag1'] = nyse['ret'].shift(1)

# linear regression of model:
reg = smf.ols(formula='ret ~ ret_lag1', data=nyse)
results = reg.fit()

# squared residuals:
nyse['resid_sq'] = results.resid ** 2
nyse['resid_sq_lag1'] = nyse['resid_sq'].shift(1)

# model for squared residuals:
ARCHreg = smf.ols(formula='resid_sq ~ resid_sq_lag1', data=nyse)
results_ARCH = ARCHreg.fit()
display(results_ARCH.summary())


# In[3]:


import numpy as np
import pandas as pd
import pandas_datareader as pdr
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import yfinance as yf

# 야후에서 Kospi, S&P 주가 다운로드
assets = ['^KS11', '^GSPC']
start = '2010-01-01'
end = '2024-6-30'

data = yf.download(assets, start=start, end=end)  
data = data.loc[:, ('Adj Close')]  
data.columns = assets  
data.columns = ['KOSPI','SNP500']
display(data)


# In[4]:


# Kospi 주가 수익율
data['ret'] = np.log(data['KOSPI']).diff()
data['ret_lag1'] = data['ret'].shift(1)
plt.plot('ret', data=data, color='black', linestyle='-')

# AR(1) model for returns:
reg = smf.ols(formula='ret ~ ret_lag1', data=data)
results = reg.fit()

# squared residuals:
data['resid_sq'] = results.resid ** 2
data['resid_sq_lag1'] = data['resid_sq'].shift(1)

# model for squared residuals:
ARCHreg = smf.ols(formula='resid_sq ~ resid_sq_lag1', data=data)
results_ARCH = ARCHreg.fit()
print(results_ARCH.summary())


# In[42]:


import wooldridge as woo
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

phillips = woo.dataWoo('phillips')
T = len(phillips)

# 1948년 이후
date_range = pd.date_range(start='1948', periods=T, freq='Y')
phillips.index = date_range.year

# Phillips curve models:
yt96 = (phillips['year'] <= 1996)
phillips['inf_diff1'] = phillips['inf'].diff()

reg_s = smf.ols(formula='Q("inf") ~ unem', data=phillips, subset=yt96)
reg_ea = smf.ols(formula='inf_diff1 ~ unem', data=phillips, subset=yt96)

results_s = reg_s.fit()
display(results_s.summary())

results_ea = reg_ea.fit()
display(results_ea.summary())

# DW tests:
DW_s = sm.stats.stattools.durbin_watson(results_s.resid)
DW_ea = sm.stats.stattools.durbin_watson(results_ea.resid)

print(f'DW_s: {DW_s}\n')
print(f'DW_ea: {DW_ea}\n')

