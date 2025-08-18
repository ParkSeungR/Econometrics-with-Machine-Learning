#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[36]:


exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[9]:


import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

# 데이터 읽어들이기
mroz = woo.dataWoo('mroz')
equation = 'inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6'

# 선형확률모형(linear probability model)의 추정
reg_lin = smf.ols(formula=equation, data=mroz)
results_lin = reg_lin.fit(cov_type='HC3')
print(results_lin.summary())

# 로짓모형(logit model)의 추정
reg_logit = smf.logit(formula= equation, data=mroz)
results_logit = reg_logit.fit()
print(results_logit.summary())

# 프로빗 모형(probit model)
reg_probit = smf.probit(formula=equation, data=mroz)
results_probit = reg_probit.fit(disp=0)
print(results_probit.summary())


# In[11]:


# 전반적 유의성 검정(test of overall significance)
llr1_manual = 2 * (results_probit.llf - results_probit.llnull)
print(f'llr1_manual: {llr1_manual}\n')
print(f'results_probit.llr: {results_probit.llr}\n')
print(f'results_probit.llr_pvalue: {results_probit.llr_pvalue}\n')

# 제약조건의 검정을 위한 왈드테스트(Wald test)
# H0: experience and age 변수는 무관한 변수
hypotheses = ['exper=0', 'I(exper ** 2)=0', 'age=0']
waldstat = results_probit.wald_test(hypotheses)
teststat2_autom = waldstat.statistic
pval2_autom = waldstat.pvalue
print(f'teststat2_autom: {teststat2_autom}\n')
print(f'pval2_autom: {pval2_autom}\n')


# In[13]:


# 특별한 관측치 정의후 예측
X_new = pd.DataFrame(
    {'nwifeinc': [100, 0], 'educ': [5, 17], 'exper': [0, 30], 
     'age': [20, 52], 'kidslt6': [2, 0], 'kidsge6': [0, 0]})
predictions_lin = results_lin.predict(X_new)
predictions_logit = results_logit.predict(X_new)
predictions_probit = results_probit.predict(X_new)

print(f'predictions_lin: \n{predictions_lin}\n')
print(f'predictions_logit: \n{predictions_logit}\n')
print(f'predictions_probit: \n{predictions_probit}\n')


# In[16]:


# 평균 한계효과의 측정 (manual average partial effects)
APE_lin = np.array(results_lin.params)

xb_logit = results_logit.fittedvalues
factor_logit = np.mean(stats.logistic.pdf(xb_logit))
APE_logit_manual = results_logit.params * factor_logit

xb_probit = results_probit.fittedvalues
factor_probit = np.mean(stats.norm.pdf(xb_probit))
APE_probit_manual = results_probit.params * factor_probit

table_manual = pd.DataFrame({'APE_lin': np.round(APE_lin, 4),
                             'APE_logit_manual': np.round(APE_logit_manual, 4),
                             'APE_probit_manual': np.round(APE_probit_manual, 4)})
print(f'table_manual: \n{table_manual}\n')

# 함수를 이용한 평균한계효과의 측정
coef_names = np.array(results_lin.model.exog_names)
coef_names = np.delete(coef_names, 0)  # drop Intercept

APE_logit_autom = results_logit.get_margeff().margeff
APE_probit_autom = results_probit.get_margeff().margeff

table_auto = pd.DataFrame({'coef_names': coef_names,
                           'APE_logit_autom': np.round(APE_logit_autom, 4),
                           'APE_probit_autom': np.round(APE_probit_autom, 4)})
print(f'table_auto: \n{table_auto}\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import scipy.stats as stats

# Random seed:
np.random.seed(1234567)

y = stats.binom.rvs(1, 0.5, size=100)
X = stats.norm.rvs(0, 1, size=100) + 2 * y
sim_data = pd.DataFrame({'y': y, 'X': X})

# 선형확률 모형의 추정
reg_lin = smf.ols(formula='y ~ X', data=sim_data)
results_lin = reg_lin.fit()
print(results_lin.summary())

#, 로짓모형의 추정
reg_logit = smf.logit(formula='y ~ X', data=sim_data)
results_logit = reg_logit.fit(disp=0)
print(results_logit.summary())

# 프로빗 모형의 추정
reg_probit = smf.probit(formula='y ~ X', data=sim_data)
results_probit = reg_probit.fit(disp=0)
print(results_probit.summary())


# In[19]:


# 한계효과(partial effects)의 계산
PE_lin = np.repeat(results_lin.params['X'], 100)

Xb_logit = results_logit.fittedvalues
factor_logit = stats.logistic.pdf(Xb_logit)
PE_logit = results_logit.params['X'] * factor_logit

Xb_probit = results_probit.fittedvalues
factor_probit = stats.norm.pdf(Xb_probit)
PE_probit = results_probit.params['X'] * factor_probit
Partial1 = pd.DataFrame(data = [X, PE_lin, PE_logit, PE_probit], index = ["X", "LPM", "Logit", "Probit"]).T
print(Partial1)


# In[23]:


Partial2 = Partial1.sort_values(by='X', ascending=True)

plt.figure(figsize =(8, 5))
plt.plot(Partial2['X'], Partial2['LPM'], linestyle='-.', color='black', label='선형확률 모형(LPM Model)')
plt.plot(Partial2['X'], Partial2['Logit'], linestyle='--', color='black', label='로짓 모형(Logit Model)')
plt.plot(Partial2['X'], Partial2['Probit'], linestyle=':', color='black', label='프로빗 모형(Probit Model)')
plt.ylabel('한계효과(partial effects)')
plt.xlabel('X')
plt.legend()
plt.show()


# In[25]:


# 예측(prediction)
X_new = pd.DataFrame({'X': np.linspace(min(X), max(X), 50)})
predictions_lin = results_lin.predict(X_new)
predictions_logit = results_logit.predict(X_new)
predictions_probit = results_probit.predict(X_new)


# In[27]:


# 그래프 그리기
plt.figure(figsize =(10, 6))
plt.plot(X, y, color='grey', marker='o', linestyle='', label='실제 데이터(actual data point)')
plt.plot(X_new['X'], predictions_lin,
         color='black', linestyle='-.', label='선형확률모형(LPM)')
plt.plot(X_new['X'], predictions_logit,
         color='black', linestyle='-', linewidth=0.5, label='로짓모형(Logit)')
plt.plot(X_new['X'], predictions_probit,
         color='black', linestyle='--', label='프로빗모형(Probit)')
plt.ylabel('y')
plt.xlabel('X')
plt.legend(loc = 'upper left')
#plt.savefig('PyGraphs/Binary-Predictions.pdf')


# In[49]:


import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

mroz = woo.dataWoo('mroz')

# LPM(linear probability model)모형 추정
reg_lin = smf.ols(formula='inlf ~ nwifeinc + educ + exper +'
                          'I(exper**2) + age + kidslt6 + kidsge6',
                  data=mroz)
results_lin = reg_lin.fit(cov_type='HC3')
display(results_lin.summary())

# 2명의 극단적 표본(two "extreme" women)에 대한 예측
X_new = pd.DataFrame(
    {'nwifeinc': [100, 0], 'educ': [5, 17],
     'exper': [0, 30], 'age': [20, 52],
     'kidslt6': [2, 0], 'kidsge6': [0, 0]})
predictions = results_lin.predict(X_new)

print(f'predictions: \n{predictions}\n')


# In[50]:


import wooldridge as woo
import statsmodels.formula.api as smf

mroz = woo.dataWoo('mroz')

# 로짓모형 추정
reg_logit = smf.logit(formula='inlf ~ nwifeinc + educ + exper +'
                              'I(exper**2) + age + kidslt6 + kidsge6',
                      data=mroz)

results_logit = reg_logit.fit(disp=10)
display(results_logit.summary())

# log likelihood value
print(f'results_logit.llf: {results_logit.llf}\n')

# McFadden's pseudo R2
print(f'results_logit.prsquared: {results_logit.prsquared}\n')


# In[51]:


import wooldridge as woo
import statsmodels.formula.api as smf

mroz = woo.dataWoo('mroz')

# 프로빗 모형 추정
reg_probit = smf.probit(formula='inlf ~ nwifeinc + educ + exper +'
                                'I(exper**2) + age + kidslt6 + kidsge6',
                        data=mroz)
results_probit = reg_probit.fit(disp=0)
display(results_probit.summary())

# log likelihood value:
print(f'results_probit.llf: {results_probit.llf}\n')
# McFadden's pseudo R2:
print(f'results_probit.prsquared: {results_probit.prsquared}\n')


# In[4]:


import wooldridge as woo
import statsmodels.formula.api as smf
import scipy.stats as stats

mroz = woo.dataWoo('mroz')

# 프로빗 모형의 추정
reg_probit = smf.probit(formula='inlf ~ nwifeinc + educ + exper +'
                                'I(exper**2) + age + kidslt6 + kidsge6',
                        data=mroz)
results_probit = reg_probit.fit(disp=20)

# 전반적 유의성 검정(overall significance test)
llr1_manual = 2 * (results_probit.llf - results_probit.llnull)
print(f'llr1_manual: {llr1_manual}\n')
print(f'results_probit.llr: {results_probit.llr}\n')
print(f'results_probit.llr_pvalue: {results_probit.llr_pvalue}\n')

# 왈드 테스트(Wald test)를 이용한 제약조건(experience와 age변수가 무의미하다는 것 테스트)
hypotheses = ['exper=0', 'I(exper ** 2)=0', 'age=0']
waldstat = results_probit.wald_test(hypotheses)
teststat2_autom = waldstat.statistic
pval2_autom = waldstat.pvalue
print(f'teststat2_autom: {teststat2_autom}\n')
print(f'pval2_autom: {pval2_autom}\n')

# 수식을 이용한 방법
reg_probit_restr = smf.probit(formula='inlf ~ nwifeinc + educ +'
                                      'kidslt6 + kidsge6',
                              data=mroz)
results_probit_restr = reg_probit_restr.fit(disp=0)

llr2_manual = 2 * (results_probit.llf - results_probit_restr.llf)
pval2_manual = 1 - stats.chi2.cdf(llr2_manual, 3)
print(f'llr2_manual2: {llr2_manual}\n')
print(f'pval2_manual2: {pval2_manual}\n')


# In[5]:


import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

mroz = woo.dataWoo('mroz')

# LPM 모형 추정
reg_lin = smf.ols(formula='inlf ~ nwifeinc + educ + exper +'
                          'I(exper**2) + age + kidslt6 + kidsge6',
                  data=mroz)
results_lin = reg_lin.fit(cov_type='HC3')

# 로짓 모형 추정
reg_logit = smf.logit(formula='inlf ~ nwifeinc + educ + exper +'
                              'I(exper**2) + age + kidslt6 + kidsge6',
                      data=mroz)
results_logit = reg_logit.fit(disp=0)

# 프로빗 모형 추정
reg_probit = smf.probit(formula='inlf ~ nwifeinc + educ + exper +'
                                'I(exper**2) + age + kidslt6 + kidsge6',
                        data=mroz)
results_probit = reg_probit.fit(disp=0)

# 극단적 관측치(two "extreme" women)에 대한 예측
X_new = pd.DataFrame(
    {'nwifeinc': [100, 0], 'educ': [5, 17],
     'exper': [0, 30], 'age': [20, 52],
     'kidslt6': [2, 0], 'kidsge6': [0, 0]})
predictions_lin = results_lin.predict(X_new)
predictions_logit = results_logit.predict(X_new)
predictions_probit = results_probit.predict(X_new)

print(f'predictions_lin: \n{predictions_lin}\n')
print(f'predictions_logit: \n{predictions_logit}\n')
print(f'predictions_probit: \n{predictions_probit}\n')


# In[6]:


import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

mroz = woo.dataWoo('mroz')

# estimate models:
reg_lin = smf.ols(formula='inlf ~ nwifeinc + educ + exper + I(exper**2) +'
                          'age + kidslt6 + kidsge6', data=mroz)
results_lin = reg_lin.fit(cov_type='HC3')

reg_logit = smf.logit(formula='inlf ~ nwifeinc + educ + exper + I(exper**2) +'
                              'age + kidslt6 + kidsge6', data=mroz)
results_logit = reg_logit.fit(disp=0)

reg_probit = smf.probit(formula='inlf ~ nwifeinc + educ + exper + I(exper**2) +'
                                'age + kidslt6 + kidsge6', data=mroz)
results_probit = reg_probit.fit(disp=0)

# 한계효과의 측정(partial effects)
APE_lin = np.array(results_lin.params)

xb_logit = results_logit.fittedvalues
factor_logit = np.mean(stats.logistic.pdf(xb_logit))
APE_logit_manual = results_logit.params * factor_logit

xb_probit = results_probit.fittedvalues
factor_probit = np.mean(stats.norm.pdf(xb_probit))
APE_probit_manual = results_probit.params * factor_probit

table_manual = pd.DataFrame({'APE_lin': np.round(APE_lin, 4),
                             'APE_logit_manual': np.round(APE_logit_manual, 4),
                             'APE_probit_manual': np.round(APE_probit_manual, 4)})
print(f'table_manual: \n{table_manual}\n')

# 함수를 이용한 한계효과 측정(automatic average partial effects)
coef_names = np.array(results_lin.model.exog_names)
coef_names = np.delete(coef_names, 0)  # drop Intercept

APE_logit_autom = results_logit.get_margeff().margeff
APE_probit_autom = results_probit.get_margeff().margeff

table_auto = pd.DataFrame({'coef_names': coef_names,
                           'APE_logit_autom': np.round(APE_logit_autom, 4),
                           'APE_probit_autom': np.round(APE_probit_autom, 4)})
print(f'table_auto: \n{table_auto}\n')


# In[25]:


import wooldridge as woo
import numpy as np
import patsy as pt
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.base.model as smclass

mroz = woo.dataWoo('mroz')
y, X = pt.dmatrices('hours ~ nwifeinc + educ + exper +'
                    'I(exper**2)+ age + kidslt6 + kidsge6',
                    data=mroz, return_type='dataframe')

# OLS를 이용한 Tobit 모형추정을 위한 초깃값 
reg_ols = smf.ols(formula='hours ~ nwifeinc + educ + exper + I(exper**2) +'
                          'age + kidslt6 + kidsge6', data=mroz)
results_ols = reg_ols.fit()
sigma_start = np.log(sum(results_ols.resid ** 2) / len(results_ols.resid))
params_start = np.concatenate((np.array(results_ols.params), sigma_start),
                              axis=None)

# Tobit 함수의 정의
class Tobit(smclass.GenericLikelihoodModel):
    def nloglikeobs(self, params):
        X = self.exog
        y = self.endog
        p = X.shape[1]
        beta = params[0:p]
        sigma = np.exp(params[p])
        y_hat = np.dot(X, beta)
        y_eq = (y == 0)
        y_g = (y > 0)
        ll = np.empty(len(y))
        ll[y_eq] = np.log(stats.norm.cdf(-y_hat[y_eq] / sigma))
        ll[y_g] = np.log(stats.norm.pdf((y - y_hat)[y_g] / sigma)) - np.log(sigma)
        return -ll

# Tobit모형 추정결과
reg_tobit = Tobit(endog=y, exog=X)
results_tobit = reg_tobit.fit(start_params=params_start, maxiter=10000, disp=0)
print(results_tobit.summary())



# In[22]:


import wooldridge as woo
import numpy as np
import patsy as pt
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.base.model as smclass

mroz = woo.dataWoo('mroz')
y, X = pt.dmatrices('hours ~ nwifeinc + educ + exper +'
                    'I(exper**2)+ age + kidslt6 + kidsge6',
                    data=mroz, return_type='dataframe')

# OLS를 이용한 Tobit 모형추정을 위한 초깃값 
reg_ols = smf.ols(formula='hours ~ nwifeinc + educ + exper + I(exper**2) +'
                          'age + kidslt6 + kidsge6', data=mroz)
results_ols = reg_ols.fit()
sigma_start = np.log(sum(results_ols.resid ** 2) / len(results_ols.resid))
params_start = np.concatenate((np.array(results_ols.params), sigma_start),
                              axis=None)

# 함수이용
exec(open("Functions/Tobit.py").read()) 
reg_tobit = Tobit(endog=y, exog=X)
results_tobit = reg_tobit.fit(start_params=params_start, maxiter=10000, disp=0)
print(results_tobit.summary()) 

# Truncated model
# pytruncreg 활용
result = truncreg(formula='hours ~ nwifeinc + educ + exper + '
                          'age + kidslt6 + kidsge6', data=mroz, point=0, direction='left')
print(result)


# A data.frame with 2725 observations on 16 variables:
# * narr86: # times arrested, 1986
# * nfarr86: # felony arrests, 1986
# * nparr86: # property crme arr., 1986
# * pcnv: proportion of prior convictions
# * avgsen: avg sentence length, mos.
# * tottime: time in prison since 18 (mos.)
# * ptime86: mos. in prison during 1986
# * qemp86: # quarters employed, 1986
# * inc86: legal income, 1986, $100s
# * durat: recent unemp duration
# * black: =1 if black
# * hispan: =1 if Hispanic
# * born60: =1 if born in 1960
# * pcnvsq: pcnv^2
# * pt86sq: ptime86^2
# * inc86sq: inc86^2
# 

# In[9]:


import wooldridge as woo
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

crime1 = woo.dataWoo('crime1')

# 선형모형
reg_lin = smf.ols(formula='narr86 ~ pcnv + avgsen + tottime + ptime86 +'
                          'qemp86 + inc86 + black + hispan + born60',
                  data=crime1)
results_lin = reg_lin.fit()
display(results_lin.summary())

# 포아송 모형(Poisson model)
reg_poisson = smf.poisson(formula='narr86 ~ pcnv + avgsen + tottime +'
                                  'ptime86 + qemp86 + inc86 + black +'
                                  'hispan + born60',
                          data=crime1)
results_poisson = reg_poisson.fit(disp=0)

display(results_poisson.summary())

# 음이항 모형(Negative binomial regression model)
reg_nbinorm = sm.NegativeBinomial.from_formula(formula='narr86 ~ pcnv + avgsen + tottime +'
                                  'ptime86 + qemp86 + inc86 + black +'
                                  'hispan + born60',
                          data=crime1)
results_nbinorm = reg_nbinorm.fit(disp=0)

display(results_nbinorm.summary())


# In[23]:


import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
df = woo.dataWoo('crime1')
df = sm.add_constant(df)

y = df['narr86']
X = df[['const', 'pcnv', 'avgsen', 'tottime', 'ptime86', 'qemp86', 'inc86', 'black', 'hispan', 'born60']]

# 포아송 모형(Poisson model)
reg_poisson = Poisson(endog=y, exog=X)
results_poisson = reg_poisson.fit(disp=0)
print(results_poisson.summary())

# 음이항 모형(Negative binomial regression model)
reg_nbinorm = NegativeBinomial(endog=y, exog=X)
results_nbinorm = reg_nbinorm.fit(disp=0)
print(results_nbinorm.summary())


# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# set the random seed:
np.random.seed(1234567)

x = np.sort(stats.norm.rvs(0, 1, size=100) + 4)
xb = -4 + 1 * x
y_star = xb + stats.norm.rvs(0, 1, size=100)
y = np.copy(y_star)
y[y_star < 0] = 0

# conditional means:
Eystar = xb
Ey = stats.norm.cdf(xb / 1) * xb + 1 * stats.norm.pdf(xb / 1)

# plot data and conditional means:
plt.figure(figsize =(10, 6))
plt.axhline(y=0, linewidth=0.5,
            linestyle='-', color='grey')
plt.plot(x, y_star, color='black',
         marker='o', fillstyle='none', linestyle='', label='y*')
plt.plot(x, y, color='black', marker='o', fillstyle='full',
         linestyle='', label='y')
plt.plot(x, Eystar, color='black', marker='',
         linestyle='-', label='E(y*)')
plt.plot(x, Ey, color='black', marker='',
         linestyle='--', label='E(y)')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
#plt.savefig('PyGraphs/Tobit-CondMean.pdf')


# A data.frame with 1445 observations on 18 variables:
# * black: =1 if black
# * alcohol: =1 if alcohol problems
# * drugs: =1 if drug history
# * super: =1 if release supervised
# * married: =1 if married when incarc.
# * felon: =1 if felony sentence
# * workprg: =1 if in N.C. pris. work prg.
# * property: =1 if property crime
# * person: =1 if crime against person
# * priors: # prior convictions
# * educ: years of schooling
# * rules: # rules violations in prison
# * age: in months
# * tserved: time served, rounded to months
# * follow: length follow period, months
# * durat: min(time until return, follow)
# * cens: =1 if duration right censored
# * ldurat: log(durat)
# 

# In[6]:


import wooldridge as woo
import numpy as np
import patsy as pt
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.base.model as smclass

recid = woo.dataWoo('recid')

# 절단된 회귀모형
# 절단된 샘플 더미변수(cens변수는 절단여부를 나타내는 더미변수)
censored = recid['cens'] != 0

# 종속변수와 독립변수 행렬정의
y, X = pt.dmatrices('ldurat ~ workprg + priors + tserved + felon +'
                    'alcohol + drugs + black + married + educ + age',
                    data=recid, return_type='dataframe')

# OLS추정후 파리미터와 표준오차를 초기값으로 지정
reg_ols = smf.ols(formula='ldurat ~ workprg + priors + tserved + felon +'
                          'alcohol + drugs + black + married + educ + age',
                  data=recid)
results_ols = reg_ols.fit()
sigma_start = np.log(sum(results_ols.resid ** 2) / len(results_ols.resid))
params_start = np.concatenate((np.array(results_ols.params), sigma_start),
                              axis=None)


# MLE 추정을 위한 클래스 정의
class CensReg(smclass.GenericLikelihoodModel):
    def __init__(self, endog, cens, exog):
        self.cens = cens
        super(smclass.GenericLikelihoodModel, self).__init__(endog, exog,
                                                             missing='none')

    def nloglikeobs(self, params):
        X = self.exog
        y = self.endog
        cens = self.cens
        p = X.shape[1]
        beta = params[0:p]
        sigma = np.exp(params[p])
        y_hat = np.dot(X, beta)
        ll = np.empty(len(y))
        # 검열되지 않음
        ll[~cens] = np.log(stats.norm.pdf((y - y_hat)[~cens] /
                                          sigma)) - np.log(sigma)
        # 검열됨
        ll[cens] = np.log(stats.norm.cdf(-(y - y_hat)[cens] / sigma))
        return -ll


# MLE 추정결과
reg_censReg = CensReg(endog=y, exog=X, cens=censored)
results_censReg = reg_censReg.fit(start_params=params_start,
                                  maxiter=10000, method='BFGS', disp=0)
print(results_censReg.summary())


# In[10]:


get_ipython().system('pip install pytruncreg')


# In[1]:


import wooldridge as woo
import numpy as np
import patsy as pt
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.base.model as smclass
import pytruncreg
from pytruncreg import truncreg


# In[5]:


recid = woo.dataWoo('recid')
recid.head(50)


# In[15]:


# 토빗 모형의 추정
# 검열된 샘플 더미변수 (dummy for censored observations)
censored = recid['cens'] != 0
# 종속변수와 독립변수 행렬정의
y, X = pt.dmatrices('ldurat ~ workprg + priors + tserved + felon +'
                    'alcohol + drugs + black + married + educ + age',
                    data=recid, return_type='dataframe')

# OLS추정후 파리미터와 표준오차를 초기값으로 지정
reg_ols = smf.ols(formula='ldurat ~ workprg + priors + tserved + felon +'
                          'alcohol + drugs + black + married + educ + age',
                  data=recid)
results_ols = reg_ols.fit()
sigma_start = np.log(sum(results_ols.resid ** 2) / len(results_ols.resid))
params_start = np.concatenate((np.array(results_ols.params), sigma_start),
                              axis=None)

# 함수이용
exec(open("Functions/CensReg.py").read()) 
reg_censReg = CensReg(endog=y, exog=X, cens=censored)
results_censReg = reg_censReg.fit(start_params=params_start,
                                  maxiter=10000, method='BFGS', disp=0)
print(results_censReg.summary())


# In[17]:


# pytruncreg 활용
result = truncreg(formula='ldurat ~ workprg + priors + tserved + felon + alcohol + drugs + black + married + educ + age', data=recid, point=0, direction='left')
print(result)


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.stats as stats

# random seed:
np.random.seed(1234567)

X = np.sort(stats.norm.rvs(0, 1, size=100) + 4)
y = -4 + 1 * X + stats.norm.rvs(0, 1, size=100)

# 모든 관측치(complete observations)와 관찰된 관측치(observed sample)
compl = pd.DataFrame({'X': X, 'y': y})
sample = compl.loc[y > 0]

# 관찰된 관측치에 대한 OLS(sample사용)
reg_ols = smf.ols(formula='y ~ X', data=sample)
results_ols = reg_ols.fit()
yhat_ols = results_ols.fittedvalues

# 모든 관측치에 대한 OLS(compl사용)
reg_tr = smf.ols(formula='y ~ X', data=compl)
results_tr = reg_tr.fit()
yhat_tr = results_tr.fittedvalues

# plot data and conditional means:
plt.figure(figsize =(10, 6))
plt.axhline(y=0, linewidth=0.5, linestyle='-', color='grey')
plt.plot(compl['X'], compl['y'], color='black',
         marker='o', fillstyle='none', linestyle='', label='모든 데이터(all data)')
plt.plot(sample['X'], sample['y'], color='black',
         marker='o', fillstyle='full', linestyle='', label='표본 데이터(sample data)')
plt.plot(sample['X'], yhat_ols, color='black',
         marker='', linestyle='--', label='Truncated Reg')
plt.plot(compl['X'], yhat_tr, color='black',
         marker='', linestyle='-', label=' OLS fit')
plt.ylabel('y')
plt.xlabel('X')
plt.legend()


# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.base.model as smclass

# random seed:
np.random.seed(1234567)

x = np.sort(stats.norm.rvs(0, 1, size=100) + 4)
xb = -4 + 1 * x
y_star = xb + stats.norm.rvs(0, 1, size=100)
y = np.copy(y_star)
y[y_star < 0] = 0

x_wc = pd.DataFrame({'const': 1, 'x': x})

# Tobit추정을 위한 클래스 정의
class Tobit(smclass.GenericLikelihoodModel):
    def nloglikeobs(self, params):
        X = self.exog
        y = self.endog
        p = X.shape[1]
        beta = params[0:p]
        sigma = np.exp(params[p])
        y_hat = np.dot(X, beta)
        y_eq = (y == 0)
        y_g = (y > 0)
        ll = np.empty(len(y))
        ll[y_eq] = np.log(stats.norm.cdf(-y_hat[y_eq] / sigma))
        ll[y_g] = np.log(stats.norm.pdf((y - y_hat)[y_g] / sigma)) - np.log(sigma)
        return -ll


# 추정 및 적합치 계산
reg_ols = sm.OLS(endog=y, exog=x_wc)
results_ols = reg_ols.fit()
yhat_ols = results_ols.fittedvalues

sigma_start = np.log(sum(results_ols.resid ** 2) / len(results_ols.resid))
params_start = np.concatenate((np.array(results_ols.params), sigma_start), axis=None)
reg_tobit = Tobit(endog=y, exog=x_wc)
results_tobit = reg_tobit.fit(start_params=params_start, disp=0)
yhat_tobit = np.dot(x_wc, np.transpose(results_tobit.params[0:2]))

# 데이터와 모형별 적합치 그래프
plt.figure(figsize =(10, 6))
plt.axhline(y=0, linewidth=0.5, linestyle='-', color='grey')
plt.plot(x, y_star, color='black', marker='o', fillstyle='none',
         linestyle='', label='모든 자료(all data')
plt.plot(x, y, color='black', marker='o', fillstyle='full',
         linestyle='', label='절단된 자료(truncated data)')
plt.plot(x, yhat_ols, color='black', marker='',
         linestyle='-', label='OLS fit')
plt.plot(x, yhat_tobit, color='black', marker='',
         linestyle='--', label='Tobit fit')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()


# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.base.model as smclass

# random seed:
np.random.seed(1234567)

x = np.sort(stats.norm.rvs(0, 1, size=100) + 4)
xb = -4 + 1 * x
y_star = xb + stats.norm.rvs(0, 1, size=100)
y = np.copy(y_star)
y[y_star < 0] = 0

x_wc = pd.DataFrame({'const': 1, 'x': x})


# 추정 및 적합치 계산
reg_ols = sm.OLS(endog=y, exog=x_wc)
results_ols = reg_ols.fit()
yhat_ols = results_ols.fittedvalues

sigma_start = np.log(sum(results_ols.resid ** 2) / len(results_ols.resid))
params_start = np.concatenate((np.array(results_ols.params), sigma_start), axis=None)

# 함수이용
exec(open("Functions/Tobit.py").read()) 
reg_tobit = Tobit(endog=y, exog=x_wc)
results_tobit = reg_tobit.fit(start_params=params_start, disp=0)
print(results_tobit.summary())
yhat_tobit = np.dot(x_wc, np.transpose(results_tobit.params[0:2]))

# 데이터와 모형별 적합치 그래프
plt.figure(figsize =(10, 6))
plt.axhline(y=0, linewidth=0.5, linestyle='-', color='grey')
plt.plot(x, y_star, color='black', marker='o', fillstyle='none',
         linestyle='', label='모든 자료(all data')
plt.plot(x, y, color='black', marker='o', fillstyle='full',
         linestyle='', label='절단된 자료(truncated data)')
plt.plot(x, yhat_ols, color='black', marker='',
         linestyle='-', label='Tobit fit')
plt.plot(x, yhat_tobit, color='black', marker='',
         linestyle='--', label='OLS fit')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()


# In[1]:


import wooldridge as woo
import statsmodels.formula.api as smf
import scipy.stats as stats

# Hecket의 Sample Selection Model
mroz = woo.dataWoo('mroz')

# step 1: 모든 관측치를 이용한 프로빗 모형의 추정
reg_probit = smf.probit(formula='inlf ~ educ + exper + I(exper**2) +'
                                'nwifeinc + age + kidslt6 + kidsge6',
                        data=mroz)
results_probit = reg_probit.fit(disp=0)
pred_inlf = results_probit.fittedvalues
mroz['inv_mills'] = stats.norm.pdf(pred_inlf) / stats.norm.cdf(pred_inlf)
print(mroz['inv_mills'])

# step 2: y를 X와 Inverse Mill's ratio에 대해 회귀분석
reg_heckit = smf.ols(formula='lwage ~ educ + exper + I(exper**2) + inv_mills',
                     subset=(mroz['inlf'] == 1), data=mroz)
results_heckit = reg_heckit.fit()

print(results_heckit.summary())



# In[2]:


import wooldridge as woo
import statsmodels.formula.api as smf
import scipy.stats as stats

# Hecket의 Sample Selection Model
mroz = woo.dataWoo('mroz')
#print(mroz)

df = mroz[['lwage', 'inlf', 'educ', 'exper', 'nwifeinc', 'age', 'kidslt6']]
y=df['lwage']
X=df[['educ', 'exper']]
X=sm.add_constant(X)
W=df[['educ', 'exper', 'nwifeinc', 'age', 'kidslt6']]
W=sm.add_constant(W)

# 1) 수식이용
# step 1: 모든 관측치를 이용한 프로빗 모형의 추정
reg_probit = smf.probit(formula='inlf ~ educ + exper +'
                                'nwifeinc + age + kidslt6',
                        data=mroz)
results_probit = reg_probit.fit(disp=0)
pred_inlf = results_probit.fittedvalues
mroz['inv_mills'] = stats.norm.pdf(pred_inlf) / stats.norm.cdf(pred_inlf)
print(results_probit.summary())
print(mroz['inv_mills'])

# step 2: y를 X와 Inverse Mill's ratio에 대해 회귀분석
reg_heckit = smf.ols(formula='lwage ~ educ + exper + inv_mills',
                     subset=(mroz['inlf'] == 1), data=mroz)
results_heckit = reg_heckit.fit()
print(results_heckit.summary())


# 2) 함수이용
exec(open("Functions/heckman.py").read()) 

res = Heckman(y, X, W).fit(method='twostep')
print(res.summary())


# In[ ]:





# In[33]:


# 다항로짓 모형의 추정

# 데이터 읽어들이기
df = pd.read_csv('data/nels_small.csv')
display(df) 

# 다항로짓모형의 추정(설명변수는 선택자의 특징을 나타내는 자료이어야 함)
mlogit = smf.mnlogit('psechoice ~ hscath + grades + faminc + famsiz + parcoll + female + black', data=df)
results_ML = mlogit.fit()
print(results_ML.summary())

# 한계효과 계산
me = results_ML.get_margeff(at= 'mean', method = 'dydx')
print(me.summary())


# In[58]:


# 조건부로짓, 혼합로짓 모형의 추정
from statsmodels.discrete.conditional_models import ConditionalLogit

# 데이터 읽어들이기
df = pd.read_csv('data/transp.csv')
display(df) 

# 조건부 로짓모형의 추정(설명변수는 선택대안의 특징을 나타내는 자료이어야 함)
y =  df['choice']
X = df[['termtime', 'invehiclecost', 'traveltime', 'travelcost', 'air', 'train', 'bus']]
id = df['id']

cl_model = ConditionalLogit(endog = y, exog = X, groups = id)
results_CL = cl_model.fit()
print(results_CL.summary())

# 한계효과 계산 ???????????????? 찾아내어야 함????
#me = results_CL.get_margeff(at= 'mean', method = 'dydx')
#print(me.summary())


# In[43]:


# 혼합로짓모형(설명변수는  선택자 & 선택대안의 특징적 자료가 혼합)
df['incair'] =  df['income']* df['air']
df['inctra'] = df['income']* df['train']
df['incbus'] = df['income']* df['bus']
df['parair'] = df['partysize']* df['air']
df['partra'] = df['partysize']* df['train']
df['parbus'] = df['partysize']* df['bus']

# 조건부 로짓모형의 추정()
y =  df['choice']
X = df[['termtime', 'invehiclecost', 'traveltime', 'travelcost', 'air', 'train', 'bus',
       'incair', 'inctra', 'incbus', 'parair', 'partra', 'parbus']]
id = df['id']

mx_model = ConditionalLogit(endog = y, exog = X, groups = id)
results_MX = mx_model.fit()
print(results_MX.summary())


# In[46]:


# 서열로짓, 서열프로빗 모형의 추정
import numpy as np
import pandas as pd
import scipy.stats as stats

from statsmodels.miscmodels.ordinal_model import OrderedModel

# 데이터 읽어들이기
df = pd.read_csv('data/warm.csv')
display(df) 

y =  df['warm']
X = df[['yr89', 'male', 'white', 'age', 'ed', 'prst']]
 
mod_prob = OrderedModel(endog=y, exog = X, distr='probit')
res_prob = mod_prob.fit(method='bfgs')
print(res_prob.summary())

mod_logit = OrderedModel(endog=y, exog = X, distr='logit')
res_logit = mod_logit.fit(method='bfgs')
print(res_logit.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




