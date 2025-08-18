#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# # PART 3-2: 다중회귀(Multiple Regression)

# In[2]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()

get_ipython().system('pip install --upgrade pandas')


# In[4]:


exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# ### 대학성적결정요인:  29 변수 141개 관측치
# * age: in years
# * soph: =1 if sophomore
# * junior: =1 if junior
# * senior: =1 if senior
# * senior5: =1 if fifth year senior
# * male: =1 if male
# * campus: =1 if live on campus
# 68 gpa1
# * business: =1 if business major
# * engineer: =1 if engineering major
# * colGPA: MSU GPA
# * hsGPA: high school GPA
# * ACT: ’achievement’ score
# * job19: =1 if job <= 19 hours
# * job20: =1 if job >= 20 hours
# * drive: =1 if drive to campus
# * bike: =1 if bicycle to campus
# * walk: =1 if walk to campus
# * voluntr: =1 if do volunteer work
# * PC: =1 of pers computer at sch
# * greek: =1 if fraternity or sorority
# * car: =1 if own car
# * siblings: =1 if have siblings
# * bgfriend: =1 if boy- or girlfriend
# * clubs: =1 if belong to MSU club
# * skipped: avg lectures missed per week
# * alcohol: avg # days per week drink alc.
# * gradMI: =1 if Michigan high school
# * fathcoll: =1 if father college grad
# * mothcoll: =1 if mother college grad
# 

# In[2]:


import wooldridge as woo
import statsmodels.formula.api as smf
import numpy as np

# 대학성적 결정요인(대학성적=F(고등학교 성적, 성과측정 점수))
gpa1 = woo.dataWoo('gpa1')

reg = smf.ols(formula='colGPA ~ hsGPA + ACT', data=gpa1)
results = reg.fit()

display(results.summary())


# In[3]:


# 이상은 다음과 같은 2, 1줄 명령어로 대체 가능함
reg = smf.ols(formula='colGPA ~ hsGPA + ACT', data=gpa1).fit().summary()
display(reg)


# In[4]:


display(smf.ols(formula='colGPA ~ hsGPA + ACT', data=gpa1).fit().summary())


# #### 임금결정요인: 24개 변수, 526개 관측치
# * wage: average hourly earnings
# * educ: years of education
# * exper: years potential experience
# * tenure: years with current employer
# * nonwhite: =1 if nonwhite
# * female: =1 if female
# * married: =1 if married
# * numdep: number of dependents
# * smsa: =1 if live in SMSA
# * northcen: =1 if live in north central U.S
# * south: =1 if live in southern region
# * west: =1 if live in western region
# * construc: =1 if work in construc. indus.
# * ndurman: =1 if in nondur. manuf. indus.
# * trcommpu: =1 if in trans, commun, pub ut
# * trade: =1 if in wholesale or retail
# * services: =1 if in services indus.
# * profserv: =1 if in prof. serv. indus.
# * profocc: =1 if in profess. occupation
# * clerocc: =1 if in clerical occupation
# * servocc: =1 if in service occupation
# * lwage: log(wage)
# * expersq: exper^2
# * tenursq: tenure^2
# 

# In[5]:


# 임금결정요인( 교육수준, 경험, 테뉴어)
wage1 = woo.dataWoo('wage1')

reg = smf.ols(formula='np.log(wage) ~ educ + exper + tenure', data=wage1)
results = reg.fit()
#print(results.summary())
display(results.summary())


# ### 401K 플랜 참가율 결정요인: 8개변수 1534개 관측치
# * prate: participation rate, percent
# * mrate: 401k plan match rate
# * totpart: total 401k participants
# * totelg: total eligible for 401k plan
# * age: age of 401k plan
# * totemp: total number of firm employees
# * sole: = 1 if 401k is firm’s sole plan
# * ltotemp: log of totemp
# 

# In[9]:


k401k = woo.dataWoo('401k')

reg = smf.ols(formula='prate ~ mrate + age', data=k401k)
results = reg.fit()
#print(results.summary())
display(results.summary())


# In[10]:


crime1 = woo.dataWoo('crime1')

# model without avgsen:
reg = smf.ols(formula='narr86 ~ pcnv + ptime86 + qemp86', data=crime1)
results = reg.fit()
#print(results.summary())
display(results.summary())


# In[11]:


# model with avgsen:
reg = smf.ols(formula='narr86 ~ pcnv + avgsen + ptime86 + qemp86', data=crime1)
results = reg.fit()
#print(results.summary())
display(results.summary())


# In[12]:


wage1 = woo.dataWoo('wage1')

reg = smf.ols(formula='np.log(wage) ~ educ', data=wage1)
results = reg.fit()
#print(results.summary())
display(results.summary())


# In[6]:


# ########################################
# 행렬연산에 의한 다중회귀 모형 추정치 ###
# ########################################

import wooldridge as woo
import numpy as np
import pandas as pd
import patsy as pt

gpa1 = woo.dataWoo('gpa1')

# 샘플 사이즈와 독립변수의 수
n = len(gpa1)
k = 2

# 종속변수(y)
y = gpa1['colGPA']

# 독립변수 행렬 X에 포함될 변수와 상수항 
X = pd.DataFrame({'const': 1, 'hsGPA': gpa1['hsGPA'], 'ACT': gpa1['ACT']})
print(y, X)


# In[7]:


# patsty 모듈을 이용한 종속변수, 독립변수 행렬 만들기(편리한 행렬만들기 방법) 
y2, X2 = pt.dmatrices('colGPA ~ hsGPA + ACT', data=gpa1, return_type='dataframe')
print(y2, X2)


# In[8]:


# 다중회귀 파라미터 추정: 행렬을 배열(array)로 만듬
X = np.array(X)
y = np.array(y)
print(y, X)


# In[9]:


# 행렬연산은 배열(array)을 이용함
b = np.linalg.inv(X.T @ X) @ X.T @ y
print(f'b: \n{b}\n')


# In[10]:


# 잔차, 잔차의 분산, 표준오차 구하기 
u_hat = y - X @ b
sigsq_hat = (u_hat.T @ u_hat) / (n - k - 1)
ser = np.sqrt(sigsq_hat)
print(f'SER: {ser}\n')


# In[11]:


# 파라미터 추정치의 분산과 표준오차 구하기 
Vbeta_hat = sigsq_hat * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diagonal(Vbeta_hat))
print(f'se: {se}\n')


# In[12]:


import wooldridge as woo
import statsmodels.formula.api as smf
import numpy as np

# 대학성적 결정요인(대학성적=F(고등학교 성적, 성과측정 점수))
gpa1 = woo.dataWoo('gpa1')

reg = smf.ols(formula='colGPA ~ hsGPA + ACT', data=gpa1)
results = reg.fit()
display(results.summary())

# 이상은 다음과 같은 2줄 또는 1줄 명령어로 대체 가능함
reg = smf.ols(formula='colGPA ~ hsGPA + ACT', data=gpa1).fit().summary()
display(reg)

display(smf.ols(formula='colGPA ~ hsGPA + ACT', data=gpa1).fit().summary())


# In[13]:


# ########################################
# 타조건불변(ceteris paribus)의 의미와 ###
# 누락변수(Omitted Variable)의 문제    ###
# ########################################

import wooldridge as woo
import statsmodels.formula.api as smf

gpa1 = woo.dataWoo('gpa1')

# 전체 모형
reg = smf.ols(formula='colGPA ~ ACT + hsGPA', data=gpa1)
results = reg.fit()
b = results.params
print(results.summary().tables[1])


# In[14]:


# relation between regressors:
reg2 = smf.ols(formula='hsGPA ~ ACT', data=gpa1)
results2 = reg2.fit()
delta = results2.params
print(results2.summary().tables[1])


# In[15]:


# omitted variables formula for b1_tilde:
b1_tilde = b['ACT'] + b['hsGPA'] * delta['ACT']
print(f'b1_tilde:  \n{b1_tilde}\n')


# In[16]:


# hsGPA 누락모형
reg3 = smf.ols(formula='colGPA ~ ACT', data=gpa1)
results3 = reg3.fit()
b_om = results3.params
print(results3.summary().tables[1])


# In[10]:


# ################################################
# 관련없는 변수 포함 문제, 통제변수 제외의 문제 ##
# ################################################
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# 데이터 읽어오기
edu_inc = pd.read_csv('Data/edu_inc.csv')
display(edu_inc)

# 상관계수 행렬
corEdu = edu_inc.corr()
print(corEdu)


# In[12]:


# 1) 누락된 변수 모형의 문제                      
# 참모형(true model)
mod1 = smf.ols('np.log(faminc) ~ hedu + wedu', data = edu_inc).fit()
print(mod1.summary().tables[1]) 
# 누락된 변수가 있는 모형
mod2 = smf.ols('np.log(faminc) ~ hedu', data = edu_inc).fit() 
print(mod2.summary().tables[1]) 

# 2) 관련없는 변수 포함모형의 문제
# 참모형
mod3 = smf.ols('np.log(faminc) ~ hedu + wedu + kl6', data = edu_inc).fit()
print(mod3.summary().tables[1])

# 관련없는 변수 포함모형
mod4 = smf.ols('np.log(faminc) ~ hedu + wedu + kl6 + xtra_x5 + xtra_x6', data=edu_inc).fit() 
print(mod4.summary().tables[1])


# In[8]:


# #########################
# 다중공선성 검정과 VIF ###
# #########################

import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

gpa1 = woo.dataWoo('gpa1')
display(gpa1)
display(gpa1.describe().T)

# 전체 모형 추정
reg = smf.ols(formula='colGPA ~ hsGPA + ACT', data=gpa1)
results = reg.fit()
print(results.summary().tables[1])

# hsGPA를 ACT 에 대해 회귀분석, R2와 VIF계산
reg2 = smf.ols(formula='hsGPA ~ ACT', data=gpa1)
results2 = reg2.fit()
R2 = results2.rsquared
VIF = 1 / (1 - R2)
print(f'\n VIF: {VIF}\n')


# In[15]:


# #########################
# 다중공선성 검정과 VIF ###
# #########################

import wooldridge as woo
import numpy as np
import statsmodels.stats.outliers_influence as smo
import patsy as pt

wage1 = woo.dataWoo('wage1')
df = wage1[['wage', 'educ', 'exper', 'tenure', 'numdep']]
display(df)
display(df.describe().T)

# 변수들 간 상관관계
print(df.corr())

# 모형 추정
reg = smf.ols(formula='np.log(wage) ~ educ + exper + tenure + numdep', data=df)
results = reg.fit()
print(results.summary().tables[1])

# 1) 수식에 의한 VIF계산법
# tenure를 educ, exper, numdep대해 회귀분석하고 R2와 VIF계산
reg1 = smf.ols(formula='tenure ~ educ + exper + numdep', data=df)
results1 = reg1.fit()
R2 = results1.rsquared
VIF = 1 / (1 - R2)
print(f'\n VIF: {VIF}\n')


# statsmodels의 VIF함수를 이용하는 방법
# 행렬 정의
y, X = pt.dmatrices('np.log(wage) ~ educ + exper + tenure + numdep',
                    data=df, return_type='dataframe')

# VIF계산
K = X.shape[1]
VIF = np.empty(K)
for i in range(K):
    VIF[i] = smo.variance_inflation_factor(X.values, i)
print(f'VIF: \n{VIF}\n')



# ## 다중회귀분석과 추론(inference)

# In[18]:


import scipy.stats as stats
import numpy as np

# 유의수준, alpha=5%, 1%, 자유도 522에서 t-분포의 임계치
alpha = np.array([0.05, 0.01])
cv_t = stats.t.ppf(1 - alpha, 522)
print(f'cv_t: {cv_t}\n')

# 유의수준, alpha=5%, 1%에서 정규분포의 임계치
cv_n = stats.norm.ppf(1 - alpha)
print(f'cv_n: {cv_n}\n')


# In[15]:


import wooldridge as woo
import statsmodels.formula.api as smf
import scipy.stats as stats

gpa1 = woo.dataWoo('gpa1')

# 모형추정
reg = smf.ols(formula='colGPA ~ hsGPA + ACT + skipped', data=gpa1)
results = reg.fit()
display(results.summary())

# 잔차 
residuals = results.resid

# Shapiro-Wilk test
SW = stats.shapiro(residuals)
print(SW)

# Jarque-Bera test
JB = stats.jarque_bera(residuals)
print(JB)

# 잔차의 히스토 그램
sns.displot(residuals, kde=True, bins=30)


# In[14]:


# 수식에 의한 t값과 p값
b = results.params
se = results.bse
tstat = b / se
pval = 2 * stats.t.cdf(-abs(tstat), 137)
print(f'tstat: \n{tstat}\n')
print(f'pval: \n{pval}\n')


# In[20]:


import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

wage1 = woo.dataWoo('wage1')

reg = smf.ols(formula='np.log(wage) ~ educ + exper + tenure', data=wage1)
results = reg.fit()
print(results.summary())


# ### R&D와 기업규모: 8개변수, 32개 관측치
# * rd: R&D spending, millions
# * sales: firm sales, millions
# * profits: profits, millions
# * rdintens: rd as percent of sales
# * profmarg: profits as percent of sales
# * salessq: sales^2
# * lsales: log(sales)
# * lrd: log(rd)
# 

# In[21]:


import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

rdchem = woo.dataWoo('rdchem')

# OLS regression:
reg = smf.ols(formula='np.log(rd) ~ np.log(sales) + profmarg', data=rdchem)
results = reg.fit()
print(results.summary())

# 95%, 99% 신뢰구간(CI)
CI95 = results.conf_int(0.05)
CI99 = results.conf_int(0.01)
print(f'CI95: \n{CI95}\n')
print(f'CI99: \n{CI99}\n')


# ### 야구 선수 연봉: 47개 변수 353개 관측치
# * teamsal: team payroll
# * nl: =1 if national league
# * years: years in major leagues
# * games: career games played
# * atbats: career at bats
# * runs: career runs scored
# * hits: career hits
# * doubles: career doubles
# * triples: career triples
# * hruns: career home runs
# * rbis: career runs batted in
# * bavg: career batting average
# * bb: career walks
# * so: career strike outs
# * sbases: career stolen bases
# * fldperc: career fielding perc
# * frstbase: = 1 if first base
# * scndbase: =1 if second base
# * shrtstop: =1 if shortstop
# * thrdbase: =1 if third base
# * outfield: =1 if outfield
# * catcher: =1 if catcher
# * yrsallst: years as all-star
# * hispan: =1 if hispanic
# * black: =1 if black
# * whitepop: white pop. in city
# * blackpop: black pop. in city
# * hisppop: hispanic pop. in city
# * pcinc: city per capita income
# * gamesyr: games per year in league
# * hrunsyr: home runs per year
# * atbatsyr: at bats per year
# * allstar: perc. of years an all-star
# * slugavg: career slugging average
# 
# * rbisyr: rbis per year
# * sbasesyr: stolen bases per year
# * runsyr: runs scored per year
# * percwhte: percent white in city
# * percblck: percent black in city
# * perchisp: percent hispanic in city
# * blckpb: black*percblck
# * hispph: hispan*perchisp
# * whtepw: white*percwhte
# * blckph: black*perchisp
# * hisppb: hispan*percblck
# * lsalary: log(salary)
# 

# In[18]:


import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

mlb1 = woo.dataWoo('mlb1')
n = mlb1.shape[0]

# 무제약 회귀(unrestricted OLS regression)
reg_ur = smf.ols(
    formula='np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr',
    data=mlb1)
fit_ur = reg_ur.fit()
r2_ur = fit_ur.rsquared
print(f'r2_ur: {r2_ur}\n')

# 제약있는 회귀(restricted OLS regression)
reg_r = smf.ols(formula='np.log(salary) ~ years + gamesyr', data=mlb1)
fit_r = reg_r.fit()
r2_r = fit_r.rsquared
print(f'r2_r: {r2_r}\n')

# F-통계량(F statistic) 계산
fstat = (r2_ur - r2_r) / (1 - r2_ur) * (n - 6) / 3
print(f'fstat: {fstat}\n')

# 유의수준 alpha=1%에서 F분포의 임계치 
cv = stats.f.ppf(1 - 0.01, 3, 347)
print(f'cv: {cv}\n')

# F 값의 p-값 (p value = 1-cdf) 
fpval = 1 - stats.f.cdf(fstat, 3, 347)
print(f'fpval: {fpval}\n')

# 함수를 이용한 F test
hypotheses = ['bavg = 0', 'hrunsyr = 0', 'rbisyr = 0']
ftest = fit_ur.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue

print(f'fstat: {fstat}\n')
print(f'fpval: {fpval}\n')

# 함수를 이용한 F test: bavg = 0, hrunsyr = 2*rbisyr의 검증
hypotheses1 = ['bavg = 0', 'hrunsyr = 2*rbisyr']
ftest1 = fit_ur.f_test(hypotheses1)
fstat1 = ftest1.statistic
fpval1 = ftest1.pvalue

print(f'fstat: {fstat}\n')
print(f'fpval: {fpval}\n')


# In[17]:


# Lagrange Multiplier(LM) Test
# 1) 제약있는 회귀(restricted OLS regression)후 R2
reg_r = smf.ols(formula='np.log(salary) ~ years + gamesyr', data=mlb1)
fit_r = reg_r.fit()
r2_r = fit_r.rsquared
residuals = fit_r.resid
print(f'r2_r: {r2_r}\n')

# 2) 제약된 모형 추정결과의 잔차를 모든 독립변수에 대해 회귀분석후 R2
mlb1['residuals'] = fit_r.resid
reg_LM = smf.ols(formula='residuals ~ years + gamesyr + bavg + hrunsyr + rbisyr',
                 data=mlb1)
fit_LM = reg_LM.fit()
r2_LM = fit_LM.rsquared
print(f'r2_LM: {r2_LM}\n')

# 3) LM test statistic 계산
LM = r2_LM * fit_LM.nobs
print(f'LM: {LM}\n')

# 4) alpha=5%에서 Chi 2의 임계치 계산
cv = stats.chi2.ppf(1-0.05, 3)
print(f'cv: {cv}\n')

# 5) p value 계산
pval = 1 - stats.chi2.cdf(LM, 3)
print(f'pval: {pval}\n')


# In[20]:


import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

mlb1 = woo.dataWoo('mlb1')
n = mlb1.shape[0]

reg_ur = smf.ols(
    formula='np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr',
    data=mlb1)
fit_ur = reg_ur.fit()
display(fit_ur.summary())

# 95%, 99% 신뢰구간(CI)
CI95 = fit_ur.conf_int(0.05)
CI99 = fit_ur.conf_int(0.01)
print(f'CI95: \n{CI95}\n')
print(f'CI99: \n{CI99}\n')


# # 추정 결과표의 정리

# In[3]:


get_ipython().system('pip install stargazer')


# In[6]:


import statsmodels.formula.api as smf
import numpy as np
import wooldridge as woo
from stargazer.stargazer import Stargazer

# 데이터 불러오기
hprice1 = woo.dataWoo('hprice1')

# 모형 1: 수준 변수
reg1 = smf.ols(formula = 'price ~ lotsize + sqrft + bdrms', data=hprice1)
results1 = reg1.fit()

# 모형 2: 로그 변환 변수
reg2 = smf.ols(formula = 'price ~ np.log(lotsize) + np.log(sqrft) + bdrms', data=hprice1)
results2 = reg2.fit()

# 모형 3: 포괄적 모형
reg3 = smf.ols(formula = 'price ~ lotsize + sqrft + bdrms + np.log(lotsize) + np.log(sqrft)', data=hprice1)
results3 = reg3.fit()

# Stargazer를 이용한 표 생성
stargazer = Stargazer([results1, results2, results3])

# 표의 제목 설정 (선택사항)
stargazer.title("Housing Price Regression Results")

# text 형식
display(stargazer)

# html 형식
result_html = stargazer.render_html()
display(result_html)

# Latex 형식
result_latex = stargazer.render_latex()
display(result_latex)


# # 다중회귀에서 몬테칼로 시뮬레이션 사례

# In[57]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(123456)

n = 100
r = 1000

beta0 = 3
beta1 = 0.4

b1 = np.empty(r)

x = stats.norm.rvs(4, 2, size=n)
for i in range(r):
    u = stats.norm.rvs(0, 1, size=n)
    y = beta0 + beta1 * x + u
    df = pd.DataFrame({'y': y, 'x': x})

    reg = smf.ols(formula='y ~ x', data=df)
    results = reg.fit()
    b1[i] = results.params['x']
    
print(b1)


# In[59]:


import seaborn as sns 
sns.displot(b1, kde=True, bins=20)


# In[6]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

# 난수의 시드(random seed) 부여
np.random.seed(123456)

# 샘플 사이즈(sample size, n)와 반복(simulations, r)횟수 설정 
n = [5, 10, 100, 1000]
r = 10000

# 모수(true parameters)값 설정
beta0 = 3
beta1 = 0.4

for j in n:
    # b1 추정치 보관 장소 
    b1 = np.empty(r)
    # X 표본생성, 반복하는 동안 일정함(fixed over replications)
    X = stats.norm.rvs(4, 2, size=j)

    # r번 반복(Simulation)
    for i in range(r):
        # 오차항 u 생성(정규분포)
        u = stats.norm.rvs(0, 0.5, size=j) 
        y = beta0 + beta1 * X + u
        df = pd.DataFrame({'y': y, 'X': X})

        # X에 대해 비조건부적인 회귀식 추정(non-conditional OLS)
        reg = smf.ols(formula='y ~ X', data=df)
        results = reg.fit()
        b1[i] = results.params['X']

    # b1 추정치 분포의 KDE
    kde = sm.nonparametric.KDEUnivariate(b1)
    kde.fit()
    
    # b1 추정치의 정규분포
    X_range = np.linspace(-2, 2.5, num=100)
    y = stats.norm.pdf(X_range, np.mean(b1), np.std(b1))

    plt.plot(kde.support, kde.density, color='black', label='b1의 KDE')
    plt.plot(X_range, y, linestyle='--', color='black', label='정규분포(normal distribution)')
    plt.ylabel('density')
    plt.xlabel('')
    plt.legend()
    plt.show()


# In[7]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

# 난수의 시드(random seed) 부여
np.random.seed(123456)

# 샘플 사이즈(sample size, n)와 반복(simulations, r)횟수 설정 
n = [5, 10, 100, 1000]
r = 10000

# 모수(true parameters)값 설정
beta0 = 3
beta1 = 0.4

for j in n:
    # b1 추정치 보관 장소 
    b1 = np.empty(r)
    # X 표본생성, 반복하는 동안 일정함(fixed over replications)
    X = stats.norm.rvs(4, 2, size=j)

    # r번 반복(Simulation)
    for i in range(r):
        # 오차항 u 생성(카이자승 분포)
        u = (stats.chi2.rvs(1, size=j) - 1) / np.sqrt(2) 
        y = beta0 + beta1 * X + u
        df = pd.DataFrame({'y': y, 'X': X})

        # X에 대해 비조건부적인 회귀식 추정(non-conditional OLS)
        reg = smf.ols(formula='y ~ X', data=df)
        results = reg.fit()
        b1[i] = results.params['X']

    # b1 추정치 분포의 KDE
    kde = sm.nonparametric.KDEUnivariate(b1)
    kde.fit()
    
    # b1 추정치의 정규분포
    X_range = np.linspace(-2, 2.5, num=100)
    y = stats.norm.pdf(X_range, np.mean(b1), np.std(b1))

    plt.plot(kde.support, kde.density, color='black', label='b1의 KDE')
    plt.plot(X_range, y, linestyle='--', color='black', label='정규분포(normal distribution)')
    plt.ylabel('density')
    plt.xlabel('')
    plt.legend()
    plt.show()


# In[67]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats

# set the random seed:
np.random.seed(1234567)

# set sample size and number of simulations:
n = 100
r = 1000

# set true parameters:
beta0 = 1
beta1 = 0.5
sx = 1
ex = 4

# initialize b1 to store results later:
b1 = np.empty(r)

# draw a sample of x, fixed over replications:
x = stats.norm.rvs(ex, sx, n)

# repeat r times:
for i in range(r):
    # draw a sample of u (uniform):
    u = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    y = beta0 + beta1 * x + u
    df = pd.DataFrame({'y': y, 'x': x})

    # estimate conditional OLS:
    reg = smf.ols(formula='y ~ x', data=df)
    results = reg.fit()
    b1[i] = results.params['x']
    
sns.displot(b1, kde=True, bins=20)


# In[68]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats

# set the random seed:
np.random.seed(1234567)

# set sample size and number of simulations:
n = 100
r = 1000

# set true parameters:
beta0 = 1
beta1 = 0.5
sx = 1
ex = 4

# initialize b1 to store results later:
b1 = np.empty(r)

# draw a sample of x, fixed over replications:
x = stats.norm.rvs(ex, sx, size=n)

# repeat r times:
for i in range(r):
    # draw a sample of u (standardized chi-squared[1]):
    u = (stats.chi2.rvs(1, size=n) - 1) / np.sqrt(2)
    y = beta0 + beta1 * x + u
    df = pd.DataFrame({'y': y, 'x': x})

    # estimate conditional OLS:
    reg = smf.ols(formula='y ~ x', data=df)
    results = reg.fit()
    b1[i] = results.params['x']

sns.displot(b1, kde=True, bins=20)


# In[9]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

# 난수의 시드(random seed) 부여
np.random.seed(123456)

# 샘플 사이즈(sample size, n)와 반복(simulations, r)횟수 설정 
n = [5, 10, 100, 1000]
r = 1000

# 모수(true parameters)값 설정
beta0 = 1
beta1 = 0.5
sx = 1
ex = 4

for j in n:
    # 기울기 파라미터, b1 
    b1 = np.empty(r)
    # 독립변수 X값 n개 생성(정규분포)
    X = stats.norm.rvs(ex, sx, size=j)
    # r회 반복
    for i in range(r):
        # 정규분포 대신 카이 스퀘어 분포
        u = (stats.chi2.rvs(1, size=j) - 1) / np.sqrt(2)
        y = beta0 + beta1 * X + u
        df = pd.DataFrame({'y': y, 'X': X})
        # 회귀식 추정
        reg = smf.ols(formula='y ~ X', data=df)
        results = reg.fit()
        b1[i] = results.params['X']

    # kde
    kde = sm.nonparametric.KDEUnivariate(b1)
    kde.fit()

    # 정규분포 밀도 함수
    X_range = np.linspace(-2.5, 3, num=100)
    y = stats.norm.pdf(X_range, np.mean(b1), np.std(b1))

    plt.plot(kde.support, kde.density, color='black', label='b1의 KDE')
    plt.plot(X_range, y, linestyle='--', color='black', label='정규분포(normal distribution)')
    plt.ylabel('density')
    plt.xlabel('')
    plt.legend()
    plt.show()



# In[11]:


# Normal vs. Chisq
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# support of normal density:
x_range = np.linspace(-4, 4, num=100)

# pdf for all these values:
pdf_n = stats.norm.pdf(x_range)
pdf_c = stats.chi2.pdf(x_range * np.sqrt(2) + 1, 1)
# pdf_c = (stats.chi2.pdf(x_range,1) - 1) / np.sqrt(2)
# plot:
plt.plot(x_range, pdf_n, linestyle='-', color='black', label='standard normal')
plt.plot(x_range, pdf_c, linestyle='--', color='black', label='standardized chi squared[1]')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()


# In[13]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats

np.random.seed(123456)

n = 100
r = 1000

beta0 = 1
beta1 = 0.5
ex = 4
sx = 1

b1 = np.empty(r)

# repeat r times:
for i in range(r):
    # 반복때마다 x변수 새로 생성
    x = stats.norm.rvs(ex, sx, size=n)
    # 오차항 생성
    u = stats.norm.rvs(0, 1, size=n)
    # 종속변수 생성
    y = beta0 + beta1 * x + u
    df = pd.DataFrame({'y': y, 'x': x})

    # 모형추정
    reg = smf.ols(formula='y ~ x', data=df)
    results = reg.fit()
    b1[i] = results.params['x']

import seaborn as sns 
sns.displot(b1, kde=True, bins=20)    


# In[16]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(123456)

n = [5, 10, 100, 1000]
r = 1000

beta0 = 1
beta1 = 0.5
ex = 4
sx = 1


for j in n:
    b1 = np.empty(r)
    for i in range(r):
        # 반복때마다 x변수 새로 생성
        x = stats.norm.rvs(ex, sx, size=j)
        u = stats.norm.rvs(0, 1, size=j)
        y = beta0 + beta1 * x + u
        df = pd.DataFrame({'y': y, 'x': x})

        # Unconditional OLS:
        reg = smf.ols(formula='y ~ x', data=df)
        results = reg.fit()
        b1[i] = results.params['x']
        
    # simulated density:
    kde = sm.nonparametric.KDEUnivariate(b1)
    kde.fit()
    # normal density/ compute mu and se
    x_range = np.linspace(-2.5, 3.5, num=100)
    y = stats.norm.pdf(x_range, np.mean(b1), np.std(b1))
    # plotting:
    plt.ylim(top=3)
    plt.xlim(-2.5, 3.5)
    plt.plot(kde.support, kde.density, color='black', label='b1')
    plt.plot(x_range, y, linestyle='--', color='black', label='normal distribution')
    plt.ylabel('density')
    plt.xlabel('')
    plt.legend()
    plt.show()


# In[14]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

# 난수의 시드(random seed) 부여
np.random.seed(123456)

# 샘플 사이즈(sample size, n)와 반복(simulations, r)횟수 설정 
n = [5, 10, 100, 1000]
r = 1000

# 모수(true parameters)값 설정
beta0 = 3
beta1 = 0.4

for j in n:
    # b1 추정치 보관 장소 
    b1 = np.empty(r)

    # r번 반복(Simulation)
    for i in range(r):
        # 반복때마다 X변수 새로 생성
        X = stats.norm.rvs(4, 2, size=j)
        # 오차항 u 생성(정규분포)
        u = stats.norm.rvs(0, 0.5, size=j)
        y = beta0 + beta1 * X + u
        df = pd.DataFrame({'y': y, 'X': X})

        # X에 대해 조건부적인 회귀식 추정(conditional OLS)
        reg = smf.ols(formula='y ~ X', data=df)
        results = reg.fit()
        b1[i] = results.params['X']

    # b1 추정치 분포의 KDE
    kde = sm.nonparametric.KDEUnivariate(b1)
    kde.fit()
    
    # b1 추정치의 정규분포
    X_range = np.linspace(-2, 2.4, num=100)
    y = stats.norm.pdf(X_range, np.mean(b1), np.std(b1))

    plt.plot(kde.support, kde.density, color='black', label='b1의 KDE')
    plt.plot(X_range, y, linestyle='--', color='black', label='정규분포(Normal distribution)')
    plt.ylabel('density')
    plt.xlabel('')
    plt.legend()
    plt.show()


# In[18]:


# ################################
# Lagrange Multiplier(LM) Test ###
# ################################

import wooldridge as woo
import statsmodels.formula.api as smf
import scipy.stats as stats

crime1 = woo.dataWoo('crime1')

# 1) 제약된 모형 추정
reg_r = smf.ols(formula='narr86 ~ pcnv + ptime86 + qemp86', data=crime1)
fit_r = reg_r.fit()
r2_r = fit_r.rsquared
print(f'r2_r: {r2_r}\n')

# 2) 제약된 모형 추정결과의 R2를 모든 독립변수에 대해 회귀분석후 R2
crime1['residuals'] = fit_r.resid
reg_LM = smf.ols(formula='residuals ~ pcnv + ptime86 + qemp86 + avgsen + tottime',
                 data=crime1)
fit_LM = reg_LM.fit()
r2_LM = fit_LM.rsquared
print(f'r2_LM: {r2_LM}\n')

# 3) LM test statistic 계산
LM = r2_LM * fit_LM.nobs
print(f'LM: {LM}\n')

# 4) alpha=5%에서 Chi 2의 임계치 계산
cv = stats.chi2.ppf(1-0.05, 2)
print(f'cv: {cv}\n')

# 5) p value 계산
pval = 1 - stats.chi2.cdf(LM, 2)
print(f'pval: {pval}\n')

# 6) F-test 결과와 동일한지 비교
reg = smf.ols(formula='narr86 ~ pcnv + ptime86 + qemp86 + avgsen + tottime',
              data=crime1)
results = reg.fit()
hypotheses = ['avgsen = 0', 'tottime = 0']
ftest = results.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue
print(f'fstat: {fstat}\n')
print(f'fpval: {fpval}\n')


# # 7. 모형에서 수식의 정의

# In[11]:


import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

bwght = woo.dataWoo('bwght')

# 기본 회귀분석과 아웃풋 출력
reg = smf.ols(formula='bwght ~ cigs + faminc', data=bwght)
results = reg.fit()
# print(results.summary())
display(results.summary())

# 변수의 단위변경(수식이용)
bwght['bwght_lbs'] = bwght['bwght'] / 16
reg_lbs = smf.ols(formula='bwght_lbs ~ cigs + faminc', data=bwght)
results_lbs = reg_lbs.fit()
# print(results_lbs.summary())
display(results_lbs.summary())

#  변수의 단위변경(함수 I()이용)
reg_lbs2 = smf.ols(formula='I(bwght/16) ~ cigs + faminc', data=bwght)
results_lbs2 = reg_lbs2.fit()
display(results_lbs2.summary())

reg_packs = smf.ols(formula='bwght ~ I(cigs/20) + faminc', data=bwght)
results_packs = reg_packs.fit()
display(results_packs.summary())

# 파라미터 추정결과 Data Frame만들어 출력
table = pd.DataFrame({'b': round(results.params, 4),
                      'b_lbs': round(results_lbs.params, 4),
                      'b_lbs2': round(results_lbs2.params, 4),
                      'b_packs': round(results_packs.params, 4)})
display(table)


# In[12]:


# 종속변수 또는 독립변수의 로그변수(함수 no.log() 활용) 또는 2-3차 제곱항 추가(함수 I()사용)
import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

hprice2 = woo.dataWoo('hprice2')
n = hprice2.shape[0]

reg = smf.ols(
    formula='np.log(price) ~ np.log(nox)+np.log(dist)+rooms+I(rooms**2)+stratio',
    data=hprice2)
results = reg.fit()

# 파라미터에 0의 제약조건부여 타당성 검증을 위한 F test
hypotheses = ['rooms = 0', 'I(rooms ** 2) = 0']
ftest = results.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue

print(f'Fstat: {fstat}\n')
print(f'fpval: {fpval}\n')


# In[25]:


# 파라미터 추정결과 출력
import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

hprice2 = woo.dataWoo('hprice2')

reg = smf.ols(
    formula='np.log(price) ~ np.log(nox)+np.log(dist)+rooms+I(rooms**2)+stratio',
    data=hprice2)
results = reg.fit()
print(results.summary())

display(results.summary())

display(results.summary().tables[1])

# Data Frame을 만들어 출력
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')
display(table)


# In[26]:


# 독립변수의 교차항 활용방법
# 방법1: x1:x2 ,예) y~x1+x2+x1:x2
# 방법2: x1*x2, 예) y~x1*x2  <==> y~x1+x2+x1:x2
# 방법3: x1*(x2+x3), 예) y~x1+x2+x3+x1:x2+x1:x3

import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

attend = woo.dataWoo('attend')
n = attend.shape[0]

reg = smf.ols(formula='stndfnl ~ atndrte*priGPA + ACT + I(priGPA**2) + I(ACT**2)',
              data=attend)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')


# In[40]:


# 독립변수, atndrte의 한계효과 (priGPA의 평균, 2.59에서 측정)
b = results.params
priGPA_mean = np.mean(attend['priGPA'])
print(f'priGPA_mean: {priGPA_mean}\n')

partial_effect = b['atndrte'] + priGPA_mean * b['atndrte:priGPA']
print(f'partial_effect: {partial_effect}\n')

# 독립변수, atndrte의 한계효과(priGPA=2.59일 때) =0 검증을 위한 F test 
hypotheses = 'atndrte + 2.59 * atndrte:priGPA = 0'
ftest = results.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue

print(f'fstat: {fstat}\n')
print(f'fpval: {fpval}\n')


# In[45]:


import wooldridge as woo
import statsmodels.formula.api as smf
import pandas as pd

gpa2 = woo.dataWoo('gpa2')

reg = smf.ols(formula='colgpa ~ sat + hsperc + hsize + I(hsize**2)', data=gpa2)
results = reg.fit()

# 예측대상 독립변수
cvalues2 = pd.DataFrame({'sat': [1200, 900, 1400, 1000, 1300],
                         'hsperc': [30, 20, 5, 22, 10], 
                         'hsize': [5, 3, 6, 4, 5]},
                         index=['newPerson1', 'newPerson2', 'newPerson3', 'newPerson4', 'newPerson5'])

# 점예측(point estimates)와 95% 신뢰구간(confidence intervals) 및 예측구간(prediction intervals)
# 점예측과 동시에 예측구간까지 계산할 때는 .get_prediction()사용
colgpa_PICI_95 = results.get_prediction(cvalues2).summary_frame(alpha=0.05)
display(colgpa_PICI_95)

# 점예측(point estimates)와 99% 신뢰구간(confidence intervals) 및 예측구간(prediction intervals)
colgpa_PICI_99 = results.get_prediction(cvalues2).summary_frame(alpha=0.01)
display(colgpa_PICI_99)

# 단순히 점예측만 할 때는 .predict() 사용
colgpa_pred = results.predict(cvalues2)
display(colgpa_pred)


# In[ ]:





# In[ ]:


import wooldridge as woo
import statsmodels.formula.api as smf
import pandas as pd

gpa2 = woo.dataWoo('gpa2')

reg = smf.ols(formula='colgpa ~ sat + hsperc + hsize + I(hsize**2)', data=gpa2)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')

# generate data set containing the regressor values for predictions:
cvalues1 = pd.DataFrame({'sat': [1200], 'hsperc': [30],
                        'hsize': [5]}, index=['newPerson1'])
print(f'cvalues1: \n{cvalues1}\n')

# point estimate of prediction (cvalues1):
colgpa_pred1 = results.predict(cvalues1)
print(f'colgpa_pred1: \n{colgpa_pred1}\n')

# define three sets of regressor variables:
cvalues2 = pd.DataFrame({'sat': [1200, 900, 1400, ],
                        'hsperc': [30, 20, 5], 'hsize': [5, 3, 1]},
                       index=['newPerson1', 'newPerson2', 'newPerson3'])
print(f'cvalues2: \n{cvalues2}\n')

# point estimate of prediction (cvalues2):
colgpa_pred2 = results.predict(cvalues2)
print(f'colgpa_pred2: \n{colgpa_pred2}\n')


# In[55]:


import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

hprice2 = woo.dataWoo('hprice2')
hprice2.describe()
# repeating the regression from Example 6.2:
reg = smf.ols(
    formula='np.log(price) ~ np.log(nox)+np.log(dist)+rooms+I(rooms**2)+stratio',
    data=hprice2)
results = reg.fit()

# predictions with rooms = 4-8, all others at the sample mean:
nox_mean = np.mean(hprice2['nox'])
dist_mean = np.mean(hprice2['dist'])
stratio_mean = np.mean(hprice2['stratio'])
X = pd.DataFrame({'rooms': np.linspace(3, 10, num=8),
                  'nox': nox_mean,
                  'dist': dist_mean,
                  'stratio': stratio_mean})
print(f'X: \n{X}\n')

# calculate 95% confidence interval:
lpr_PICI = results.get_prediction(X).summary_frame(alpha=0.05)
lpr_CI = lpr_PICI[['mean', 'mean_ci_lower', 'mean_ci_upper']]
print(f'lpr_CI: \n{lpr_CI}\n')

lpr_CI = np.exp(lpr_CI)

print(lpr_CI)

# 예측의 신뢰구간
plt.plot(X['rooms'], lpr_CI['mean'], color='black',
         linestyle='-', label='예측치(Predict)')
plt.plot(X['rooms'], lpr_CI['mean_ci_upper'], color='lightgrey',
         linestyle='--', label='신뢰구간(상)(upper CI)')
plt.plot(X['rooms'], lpr_CI['mean_ci_lower'], color='darkgrey',
         linestyle='--', label='신뢰구간(하)(lower CI)')
plt.ylabel('주택가격(price)')
plt.xlabel('룸갯수(rooms)')
plt.legend()
plt.show()


# In[57]:


import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

hprice2 = woo.dataWoo('hprice2')
hprice2.describe()
# 다중회귀함수 추정
reg = smf.ols(
    formula='np.log(price) ~ np.log(nox)+np.log(dist)+rooms+I(rooms**2)+stratio',
    data=hprice2)
results = reg.fit()

# rooms변수는 3-10 구간 정의, 다른 모든 변수는 평균값을 계산 
nox_mean = np.mean(hprice2['nox'])
dist_mean = np.mean(hprice2['dist'])
stratio_mean = np.mean(hprice2['stratio'])

X = pd.DataFrame({'rooms': np.linspace(3, 9, num=8),
                  'nox': nox_mean,
                  'dist': dist_mean,
                  'stratio': stratio_mean})
print(f'X: \n{X}\n')

# 예측치와 95% 신뢰구간 계산(이때는 predict와 forcast의 개념 구분 필요)
lpr_PICI = results.get_prediction(X).summary_frame(alpha=0.05)
display(lpr_PICI)

# 종속변수가 로그변환된 것이므로 예측치는 지수변환
price_fit = np.exp(lpr_PICI)

display(price_fit)


# In[58]:


# plot:
plt.plot(X['rooms'], price_fit['mean'], color='black',
         linestyle='-', label='예측치(Predict)')
plt.plot(X['rooms'], price_fit['mean_ci_upper'], color='lightgrey',
         linestyle='--', label='Prediction 신뢰구간(상))')
plt.plot(X['rooms'], price_fit['mean_ci_lower'], color='lightgrey',
         linestyle='--', label='Prediction 신뢰구간(하)')
plt.plot(X['rooms'], price_fit['obs_ci_upper'], color='darkgrey',
         linestyle=':', label='Forecast 신뢰구간(상)')
plt.plot(X['rooms'], price_fit['obs_ci_lower'], color='darkgrey',
         linestyle=':', label='Forecast 신뢰구간(하)')
plt.scatter(hprice2['rooms'],hprice2['price'], s=5)
plt.ylabel('주택가격(price)')
plt.xlabel('룸갯수(rooms)')
plt.legend()
plt.show()


# In[59]:


# plot:
plt.plot(X['rooms'], price_fit['mean'], color='black',
         linestyle='-', label='예측치(Predict)')
plt.fill_between(X['rooms'], price_fit['obs_ci_lower'], price_fit['obs_ci_upper'],
                  color='0.9', label='Forecast 신뢰구간')
plt.fill_between(X['rooms'], price_fit['mean_ci_lower'], price_fit['mean_ci_upper'],
                  color='0.7', label='Prediction 신뢰구간')
plt.scatter(hprice2['rooms'],hprice2['price'], s=5)
plt.ylabel('주택가격(price)')
plt.xlabel('룸갯수(rooms)')
plt.legend(loc='upper left')
plt.show()


# In[ ]:




