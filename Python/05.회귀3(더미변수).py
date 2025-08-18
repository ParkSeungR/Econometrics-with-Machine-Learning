#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# # 5. 정성적 독립변수를 가진 다중회귀모형

# ## 더미변수(Dummy Variable)

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[3]:


exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[3]:


import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

wage1 = woo.dataWoo('wage1')
display(wage1)
wage1.describe().T


# In[4]:


reg = smf.ols(formula='wage ~ female + educ + exper + tenure', data=wage1)
results = reg.fit()
#print(results.summary())
display(results.summary())

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
#print(f'table: \n{table}\n')
display(table)


# In[17]:


import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

wage1 = woo.dataWoo('wage1')

# boolean variable:
wage1['dum_female'] = (wage1['female'] == 1)
display(wage1)

reg = smf.ols(formula='wage ~ dum_female + educ + exper + tenure', data=wage1)
results = reg.fit()
#print(results.summary())
display(results.summary())

# 데이터 프레임을 이용한 추정결과 출력
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
#print(f'table: \n{table}\n')
display(table)


# In[16]:


# Scatter plot with two categories. Dataset: utown
plt.figure(figsize = (5, 3))
plt.scatter('educ', 'wage', label = 'Female', color = '0.6', s=5, data = wage1[wage1['female'] == 1])
plt.scatter('educ', 'wage', label = 'Male',   color = '0.4', s=5, data = wage1[wage1['female'] == 0])
plt.xlabel('Education')
plt.ylabel('Wage')
plt.legend()
plt.tight_layout()


# In[20]:


# 더미변수 만들기
Dummy_female = pd.get_dummies(wage1, columns=['female'], drop_first=True)
print(Dummy_female)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

CPS1985 = pd.read_csv('Data/cps1985.csv')

# 카테고리 변수별(성별) 빈도수 
freq_gender = pd.crosstab(CPS1985['gender'], columns='count')
print(f'freq_gender: \n{freq_gender}\n')

# 카테고리 변수별(직업별) 빈도수 
freq_occupation = pd.crosstab(CPS1985['occupation'], columns='count')
print(f'freq_occupation: \n{freq_occupation}\n')

# 카테고리변수를 직접 회귀식 Formula의 독립변수에 이용하는 방법: C()함수 이용
reg = smf.ols(formula='np.log(wage) ~ education +'
                      'experience + C(gender) + C(occupation)', data=CPS1985)
results = reg.fit()
#print(results.summary())
display(results.summary())

# 데이터 프레임을 이용한 추정결과 출력
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
#print(f'table: \n{table}\n')
display(table)


# In[7]:


# # 카테고리변수에서 기준변수(reference)지정 방법: treatment()사용
reg_newref = smf.ols(formula='np.log(wage) ~ education + experience + '
                             'C(gender, Treatment("male")) + '
                             'C(occupation, Treatment("technical"))', data=CPS1985)
results_newref = reg_newref.fit()
#print(results_newref.summary())
display(results_newref.summary())

# print results:
table_newref = pd.DataFrame({'b': round(results_newref.params, 4),
                             'se': round(results_newref.bse, 4),
                             't': round(results_newref.tvalues, 4),
                             'pval': round(results_newref.pvalues, 4)})
#print(f'table_newref: \n{table_newref}\n')
display(table_newref)


# In[10]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

CPS1985 = pd.read_csv('Data/cps1985.csv')

# 회귀분석
reg = smf.ols(
    formula='np.log(wage) ~ education + experience + gender + occupation',
    data=CPS1985)
results = reg.fit()
display(results.summary())

# 분산분석(ANOVA)
table_anova = sm.stats.anova_lm(results, typ=2)
display(table_anova)


# In[16]:


# 연속형 변수를 카테고리 변수로 나누어 회귀분석
import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

lawsch85 = woo.dataWoo('lawsch85')

lawsch85.describe()


# In[19]:


# 연속형 변수를 나눌 구간 설정
cut_points = [1, 30, 80, 120, 175]

# 카테고리변수 생성
lawsch85['rank_c'] = pd.cut(lawsch85['rank'], bins=cut_points,
                        labels=['(1_30]', '(30_80]', '(80_120]', '(120_175]'])

# 카테고리별 빈도수
freq = pd.crosstab(lawsch85['rank_c'], columns='count')
display(freq)

# 회귀분석
reg = smf.ols(formula='np.log(salary) ~ C(rank_c, Treatment("(120_175]")) +'
                      'LSAT + GPA + np.log(libvol) + np.log(cost)',
                      data=lawsch85)
results = reg.fit()
display(results.summary())

# 분산분석표(ANOVA table)
table_anova = sm.stats.anova_lm(results, typ=2)
display(table_anova)


# In[20]:


# 카테고리 변수 가운데 일부에 대한 회귀분석(1)
import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

gpa3 = woo.dataWoo('gpa3')

# model with full interactions with female dummy (only for spring data):
reg = smf.ols(formula='cumgpa ~ female * (sat + hsperc + tothrs)',
              data=gpa3, subset=(gpa3['spring'] == 1))
results = reg.fit()
display(results.summary())

# 관심변수의 효과에 대한 F검정
hypotheses = ['female = 0', 'female:sat = 0',
              'female:hsperc = 0', 'female:tothrs = 0']
ftest = results.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue

print(f'fstat: {fstat}\n')
print(f'fpval: {fpval}\n')


# In[8]:


# 카테고리 변수 가운데 일부에 대한 회귀분석(2)
import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

gpa3 = woo.dataWoo('gpa3')

# males, spring data에 대한 회귀분석
reg_m = smf.ols(formula='cumgpa ~ sat + hsperc + tothrs',
                data=gpa3,
                subset=(gpa3['spring'] == 1) & (gpa3['female'] == 0))
results_m = reg_m.fit()
display(results.summary())

# females, spring data에 대한 회귀분석
reg_f = smf.ols(formula='cumgpa ~ sat + hsperc + tothrs',
                data=gpa3,
                subset=(gpa3['spring'] == 1) & (gpa3['female'] == 1))
results_f = reg_f.fit()
display(results_f.summary())


# In[ ]:





# In[24]:


import wooldridge as woo
import pandas as pd
import numpy as np
import scipy.stats as stats 
import statsmodels.formula.api as smf
import patsy as pt
import matplotlib.pyplot as pit 
import statsmodels.api as sm 
from statsmodels.base.model import GenericLikelihoodModel

wage1 = woo.dataWoo('wage1')

# patsty 모듈을 이용한 종속변수, 독립변수 행렬 만들기
y, X = pt.dmatrices('wage ~ C(female) + educ + exper + tenure', data=wage1, return_type='dataframe')
print(y, X)


# In[25]:


# Maximum Likelihood 
class OLSMLE(GenericLikelihoodModel): 
    def loglike(self, params): 
        exog = self.exog 
        endog = self.endog 
        k = exog.shape[1]
        resids = endog - np.dot(exog, params[0:k]) 
        sigma = np.std(resids, ddof=0)
        return stats.norm.logpdf(resids, loc=0, scale=sigma).sum()

# ML에 의한 모형추정 
resultsML = OLSMLE(y, X).fit()
print(resultsML.summary())


# In[ ]:





# In[26]:


ols_resid = sm.OLS(y, X).fit().resid
ols_resid 


# In[27]:


y_r = np.asarray(ols_resid)[1:]
X_r = np.asarray(ols_resid)[:-1]
X_r = sm.add_constant(X_r)
print(y_r, X_r)


# In[28]:


resid_fit = sm.OLS(y_r, X_r).fit()


# In[29]:


print(resid_fit.tvalues[1])
print(resid_fit.pvalues[1])


# In[30]:


rho = resid_fit.params[1]
rho


# In[31]:


from scipy.linalg import toeplitz

order = toeplitz(range(len(ols_resid)))
order


# In[32]:


sigma = rho ** order
sigma


# In[33]:


# Updating the covariance matrix for heteroscedasticity 
nobs = X.shape[0]

for i in range(0, nobs):
    for j in range(0, nobs):
        sigma[i,j] = (np.sqrt(i+1)*np.sqrt(j+1))/(1-rho**2)*sigma[i,j]

# Showing the covariance matrix 
print(np.round(sigma, 4))


# In[34]:


gls_model = sm.GLS(y, X, sigma=sigma)
gls_results = gls_model.fit()
print(gls_results.summary())


# In[ ]:





# In[35]:


glsar_model = sm.GLSAR(y, X, 1)
glsar_results = glsar_model.iterative_fit(1)
print(glsar_results.summary())


# In[ ]:





# In[52]:


resid_fgls = sm.OLS(np.log(ols_resid**2), X).fit() 
e_hat = np.sqrt(np.exp(resid_fgls.fittedvalues))

# Creating the covariance matrix based on autocorrelation 

sigma_hat = rho**order
# Updating the covariance matrix for heteroscedasticity 
for i in range(0, nobs):
    for j in range(0, nobs) :
        sigma_hat[i,j] - (e_hat[i]*e_hat[j])/(1- rho**2)*sigma_hat[i,j]

print(np.round(sigma_hat, 4))

# Calculating the coefficients using FGLS
FGLS_Result = sm.GLS(y, X, sigma=sigma_hat).fit() 
print(FGLS_Result.summary())


# In[ ]:





# In[ ]:





# In[ ]:




