#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[3]:


exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# A data.frame with 1084 observations on 15 variables:
# * educ: years of schooling
# * south: =1 if live in south
# * nonwhite: =1 if nonwhite
# * female: =1 if female
# * married: =1 if married
# * exper: age - educ - 6
# * expersq: exper^2
# * union: =1 if belong to union
# * lwage: log hourly wage
# * age: in years
# * year: 78 or 85
# * y85: =1 if year == 85
# * y85fem: y85*female
# * y85educ: y85*educ
# * y85union: y85*union
# 

# ## 교육의 임금 효과, 성별 임금격차

# In[10]:


import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

df = woo.dataWoo('cps78_85')
print(df)
print(df.describe())
print(df.info())

# 연도 더미변수와의 교차항을 포함한 OLS
reg = smf.ols(formula='lwage ~ y85*(educ+female) + exper +'
                      'I(exper**2) + union',
              data=df)
results = reg.fit()
print(results.summary())


# ## 쓰레기 처리장 위치와 주택가격

#  A data.frame with 321 observations on 25 variables:
# * year: 1978 or 1981
# * age: age of house
# * agesq: age^2
# * nbh: neighborhood, 1-6
# * cbd: dist. to cent. bus. dstrct, ft.
# * intst: dist. to interstate, ft.
# * lintst: log(intst)
# * price: selling price
# * rooms: # rooms in house
# * area: square footage of house
# * land: square footage lot
# * baths: # bathrooms
# * dist: dist. from house to incin., ft.
# * ldist: log(dist)
# * wind: prc. time wind incin. to house
# * lprice: log(price)
# * y81: =1 if year == 1981
# * larea: log(area)
# * lland: log(land)
# * y81ldist: y81*ldist
# * lintstsq: lintst^2
# * nearinc: =1 if dist <= 15840
# * y81nrinc: y81*nearinc
# * rprice: price, 1978 dollars
# * lrprice: log(rprice)
# 

# In[7]:


import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

# DID
df = woo.dataWoo('kielmc')
#print(df)
#print(df.describe())

# 1978년과 1981년 분리하여 회귀분석
df_y78 = df[df['year'] == 1978]
reg78 = smf.ols(formula='rprice ~ nearinc', data=df_y78)
results78 = reg78.fit()
print(results78.summary().tables[1])

df_y81 = df[df['year'] == 1981]
reg81 = smf.ols(formula='rprice ~ nearinc', data=df_y81)
results81 = reg81.fit()
print(results81.summary().tables[1])

# 연도 더미변수와의 교차항을 포함하는 회귀
reg_joint = smf.ols(formula='rprice ~ nearinc * C(year)', data=df)
results_joint = reg_joint.fit()
print(results_joint.summary().tables[1])


# ## Difference in Difference(DID)

# In[8]:


import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

kielmc = woo.dataWoo('kielmc')

# DiD 모형
reg_did = smf.ols(formula='lrprice ~ nearinc*C(year)', data=kielmc)
results_did = reg_did.fit()
print(results_did.summary()) 


# 통제변수(control variables)를 가진 DID
formula= """lrprice ~ nearinc*C(year) + age + I(age**2) + np.log(intst) 
          + np.log(land) + np.log(area) + rooms + baths"""
reg_didC = smf.ols(formula=formula, data=kielmc)
results_didC = reg_didC.fit()
print(results_didC.summary()) 


# ## First Difference Model

#  

# A data.frame with 106 observations on 12 variables:
# * district: district number
# * year: 72 or 78
# * crime: crimes per 1000 people
# * clrprc1: clear-up perc, prior year
# * clrprc2: clear-up perc, two-years prior
# * d78: =1 if year = 78
# * avgclr: (clrprc1 + clrprc2)/2
# * lcrime: log(crime)
# * clcrime: change in lcrime
# * cavgclr: change in avgclr
# * cclrprc1: change in clrprc1
# * cclrprc2: change in clrprc2
# 

# In[10]:


import wooldridge as woo
import numpy as np
import linearmodels as plm

crime4 = woo.dataWoo('crime4')
crime4 = crime4.set_index(['county', 'year'], drop=False)
print(crime4.describe().T)

# 1차 차분모형(FD model)
equation = """lcrmrte ~ year + d83 + d84 + d85 + d86 + d87 +
            lprbarr + lprbconv + lprbpris + lavgsen + lpolpc"""
reg = plm.FirstDifferenceOLS.from_formula(formula=equation,
      data=crime4)
results = reg.fit()

# .summary()가 제외되었음에 주의!!!
print(results)


#  

# A data.frame with 92 observations on 34 variables:
# * pop: population
# * crimes: total number index crimes
# * unem: unemployment rate
# * officers: number police officers
# * pcinc: per capita income
# * west: =1 if city in west
# * nrtheast: =1 if city in NE
# * south: =1 if city in south
# * year: 82 or 87
# * area: land area, square miles
# * d87: =1 if year = 87
# * popden: people per sq mile
# * crmrte: crimes per 1000 people
# * offarea: officers per sq mile
# * lawexpc: law enforce. expend. pc, $
# * polpc: police per 1000 people
# * lpop: log(pop)
# * loffic: log(officers)
# * lpcinc: log(pcinc)
# * llawexpc: log(lawexpc)
# * lpopden: log(popden)
# * lcrimes: log(crimes)
# * larea: log(area)
# * lcrmrte: log(crmrte)
# * clcrimes: change in lcrimes
# * clpop: change in lpop
# * clcrmrte: change in lcrmrte
# * lpolpc: log(polpc)
# * clpolpc: change in lpolpc
# * cllawexp: change in llawexp
# * cunem: change in unem
# * clpopden: change in lpopden
# * lcrmrt_1: lcrmrte lagged
# * ccrmrte: change in crmrte
# 

# ### 추가

# In[7]:


import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import linearmodels as plm

crime2 = woo.dataWoo('crime2')

# year변수로 부터 더미변수 생성
crime2['t'] = (crime2['year'] == 87).astype(int)

# 균형패널에서 id변수 만들기 
id_tmp = np.linspace(1, 46, num=46)

# 1-46개의 배열을 수직으로 합친 다음 정령하여 하나의 id에 2개년도 자료 
crime2['id'] = np.sort(np.concatenate([id_tmp, id_tmp]))
print(crime2['id'])

# crmrte과 unem에 대한 1차 차분 
crime2['crmrte_diff1'] = \
    crime2.sort_values(['id', 'year']).groupby('id')['crmrte'].diff()
crime2['unem_diff1'] = \
    crime2.sort_values(['id', 'year']).groupby('id')['unem'].diff()
var_selection = ['id', 't', 'crimes', 'unem', 'crmrte_diff1', 'unem_diff1']
display(crime2)


# In[5]:


# 차분된 자료(differenced data)로 OLS 회귀
reg_sm = smf.ols(formula='crmrte_diff1 ~ unem_diff1', data=crime2)
results_sm = reg_sm.fit()
display(results_sm.summary())

# 차분모형 추정을 위한 명령어를 활용한 모형추정
crime2 = crime2.set_index(['id', 'year'])
reg_plm = plm.FirstDifferenceOLS.from_formula(formula='crmrte ~ t + unem',
                                              data=crime2)
results_plm = reg_plm.fit()

display(results_plm)


#  

# In[1]:


import wooldridge as woo
import numpy as np
import pandas as pd
import linearmodels as plm

crime4 = woo.dataWoo('crime4')
crime4 = crime4.set_index(['county', 'year'], drop=False)

# estimate FD model:
reg = plm.FirstDifferenceOLS.from_formula(
      formula='lcrmrte ~ year + d83 + d84 + d85 + d86 + d87 +'
            'lprbarr + lprbconv + lprbpris + lavgsen + lpolpc',
      data=crime4)

# regression with standard SE:
results_default = reg.fit()
display(results_default)

# regression with "clustered" SE:
results_cluster = reg.fit(cov_type='clustered', cluster_entity=True,
                          debiased=False)
display(results_cluster)

# regression with "clustered" SE (small-sample correction):
results_css = reg.fit(cov_type='clustered', cluster_entity=True)
display(results_css)


#  

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

# In[10]:


# ################################
import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf
import linearmodels as plm

jtrain = woo.dataWoo('jtrain')
jtrain['entity'] = jtrain['fcode']
jtrain = jtrain.set_index(['fcode', 'year'])

# Manual computation of deviations of entity means:
jtrain['lscrap_w'] = jtrain['lscrap'] - jtrain.groupby('fcode').mean()['lscrap']
jtrain['d88_w'] = jtrain['d88'] - jtrain.groupby('fcode').mean()['d88']
jtrain['d89_w'] = jtrain['d89'] - jtrain.groupby('fcode').mean()['d89']
jtrain['grant_w'] = jtrain['grant'] - jtrain.groupby('fcode').mean()['grant']
jtrain['grant_1_w'] = jtrain['grant_1'] - jtrain.groupby('fcode').mean()['grant_1']

# manual FE model estimation:
results_man = smf.ols(formula='lscrap_w ~ 0 + d88_w + d89_w + grant_w + grant_1_w', data=jtrain).fit()
table_man = pd.DataFrame({'b': round(results_man.params, 4),
                          'se': round(results_man.bse, 4),
                          't': round(results_man.tvalues, 4),
                          'pval': round(results_man.pvalues, 4)})
print(f'table_man: \n{table_man}\n')

# automatic FE model estimation:
reg_aut = plm.PanelOLS.from_formula(formula='lscrap ~ d88 + d89 + grant + grant_1 + EntityEffects', data=jtrain)
results_aut = reg_aut.fit()
table_aut = pd.DataFrame({'b': round(results_aut.params, 4),
                          'se': round(results_aut.std_errors, 4),
                          't': round(results_aut.tstats, 4),
                          'pval': round(results_aut.pvalues, 4)})
print(f'table_aut: \n{table_aut}\n')


# A data.frame with 4360 observations on 44 variables:
# * nr: person identifier
# * year: 1980 to 1987
# * agric: =1 if in agriculture
# * black: =1 if black
# * bus:
# * construc: =1 if in construction
# * ent:
# * exper: labor mkt experience
# * fin:
# * hisp: =1 if Hispanic
# * poorhlth: =1 if in poor health
# * hours: annual hours worked
# * manuf: =1 if in manufacturing
# * married: =1 if married
# * min:
# * nrthcen: =1 if north central
# * nrtheast: =1 if north east
# * occ1:
# * occ2:
# * occ3:
# * occ4:
# * occ5:
# * occ6:
# * occ7:
# * occ8:
# * occ9:
# * per:
# * pro:
# * pub:
# * rur:
# * south: =1 if south
# * educ: years of schooling
# * tra:
# * trad:
# * union: =1 if in union
# * lwage: log(wage)
# * d81: =1 if year == 1981
# * d82:
# * d83:
# * d84:
# * d85:
# * d86:
# * d87:
# * expersq: exper^2
# 

#  

# ### Fixed Effects

# In[10]:


import wooldridge as woo
import pandas as pd
import linearmodels as plm

wagepan = woo.dataWoo('wagepan')
wagepan = wagepan.set_index(['nr', 'year'], drop=False)

# 고정효과 모형(FE model)
reg = plm.PanelOLS.from_formula(
      formula='lwage ~ married + union + C(year)*educ + EntityEffects',
      data=wagepan, drop_absorbed=True)
results = reg.fit()
display(results)


# In[29]:


import wooldridge as woo
import pandas as pd
import linearmodels as plm

wagepan = woo.dataWoo('wagepan')
display(wagepan)

# 패널자료의 N, T, n
N = wagepan.shape[0]
T = wagepan['year'].drop_duplicates().shape[0]
n = wagepan['nr'].drop_duplicates().shape[0]
print(N, T, n)

# 개인 불변, 시간불변 변수 확인 
results1 = wagepan.groupby('nr').var()
display(results1)

results2 = wagepan.groupby('year').var()
display(results2)


# In[33]:


import wooldridge as woo
import pandas as pd
import linearmodels as plm

wagepan = woo.dataWoo('wagepan')
display(wagepan)

wagepan = pd.DataFrame(wagepan)
wagepan = wagepan.set_index(['nr', 'year'], drop=False)
display(wagepan)

formula= """lwage ~ educ + black + hisp + exper + I(exper**2) +
                    married + union + C(year)"""

# Pooling
reg_ols = plm.PooledOLS.from_formula(formula=formula, data=wagepan)
results_ols = reg_ols.fit()
display(results_ols)

# 고정효과 모형
reg_fe = plm.PanelOLS.from_formula(
    formula='lwage ~ I(exper**2) + married + union +'
            'C(year) + EntityEffects', data=wagepan)
results_fe = reg_fe.fit()
display(results_fe)

# 확률효과 모형
reg_re = plm.RandomEffects.from_formula(formula=formula, data=wagepan)
results_re = reg_re.fit()
display(results_re)


# In[34]:


import wooldridge as woo
import pandas as pd
import linearmodels as plm
# 어떤 의미인지 잘 이해안됨
wagepan = woo.dataWoo('wagepan')
wagepan['t'] = wagepan['year']
wagepan['entity'] = wagepan['nr']
wagepan = wagepan.set_index(['nr'])

# include group specific means:
wagepan['married_b'] = wagepan.groupby('nr').mean()['married']
wagepan['union_b'] = wagepan.groupby('nr').mean()['union']
wagepan = wagepan.set_index(['year'], append=True)
display(wagepan)


# In[36]:


# estimate CRE paramters:
reg = plm.RandomEffects.from_formula(
      formula='lwage ~ married + union + educ +'
            'black + hisp + married_b + union_b',
      data=wagepan)
results = reg.fit()
display(results)


# In[37]:


import wooldridge as woo
import linearmodels as plm

wagepan = woo.dataWoo('wagepan')
wagepan['t'] = wagepan['year']
wagepan['entity'] = wagepan['nr']
wagepan = wagepan.set_index(['nr'])

# include group specific means:
wagepan['married_b'] = wagepan.groupby('nr').mean()['married']
wagepan['union_b'] = wagepan.groupby('nr').mean()['union']
wagepan = wagepan.set_index(['year'], append=True)

# estimate CRE:
reg_cre = plm.RandomEffects.from_formula(
    formula='lwage ~ married + union + C(t)*educ  + married_b + union_b',
    data=wagepan)
results_cre = reg_cre.fit()

# RE test as an Wald test on the CRE specific coefficients:
wtest = results_cre.wald_test(formula='married_b = union_b = 0')
print(f'wtest: \n{wtest}\n')


# In[16]:


import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf
import linearmodels as plm

wagepan = woo.dataWoo('wagepan')
wagepan['t'] = wagepan['year']
wagepan['entity'] = wagepan['nr']
wagepan = wagepan.set_index(['nr'])

# include group specific means:
wagepan['married_b'] = wagepan.groupby('nr').mean()['married']
wagepan['union_b'] = wagepan.groupby('nr').mean()['union']
wagepan = wagepan.set_index(['year'], append=True)

# estimate FE parameters in 3 different ways:
reg_we = plm.PanelOLS.from_formula(
    formula='lwage ~ married + union + C(t)*educ + EntityEffects',
    drop_absorbed=True, data=wagepan)
results_we = reg_we.fit()

reg_dum = smf.ols(
    formula='lwage ~ married + union + C(t)*educ + C(entity)',
    data=wagepan)
results_dum = reg_dum.fit()

reg_cre = plm.RandomEffects.from_formula(
    formula='lwage ~ married + union + C(t)*educ + married_b + union_b',
    data=wagepan)
results_cre = reg_cre.fit()

# compare to RE estimates:
reg_re = plm.RandomEffects.from_formula(
    formula='lwage ~ married + union + C(t)*educ',
    data=wagepan)
results_re = reg_re.fit()

var_selection = ['married', 'union', 'C(t)[T.1982]:educ']

# print results:
table = pd.DataFrame({'b_we': round(results_we.params[var_selection], 4),
                      'b_dum': round(results_dum.params[var_selection], 4),
                      'b_cre': round(results_cre.params[var_selection], 4),
                      'b_re': round(results_re.params[var_selection], 4)})
print(f'table: \n{table}\n')


# In[19]:


import wooldridge as woo
import numpy as np
import linearmodels as plm
import scipy.stats as stats

wagepan = woo.dataWoo('wagepan')
wagepan = wagepan.set_index(['nr', 'year'], drop=False)

# FE와 RE 추정
reg_fe = plm.PanelOLS.from_formula(formula='lwage ~ I(exper**2) + married +'
                                           'union + C(year) + EntityEffects',
                                   data=wagepan)
results_fe = reg_fe.fit()
b_fe = results_fe.params
b_fe_cov = results_fe.cov

reg_re = plm.RandomEffects.from_formula(
    formula='lwage ~ educ + black + hisp + exper + I(exper**2)'
            '+ married + union + C(year)', data=wagepan)
results_re = reg_re.fit()
b_re = results_re.params
b_re_cov = results_re.cov


# In[20]:


# 하우스만 테스트(Hausman test of FE vs. RE)
# (I) find overlapping coefficients:
common_coef = list(set(results_fe.params.index).intersection(results_re.params.index))

# (II) FE and RE 파라미터의 차이
b_diff = np.array(results_fe.params[common_coef] - results_re.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')


# In[17]:


# 하우스만 테스트(Hausman test of FE vs. RE) 함수 정의
def HausmanTest(Results_FE, Results_RE):
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
    return [Chi2, dof, pvalue]

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

H, dof, pvalue = HausmanTest(Results_FE, Results_RE)
print(H, dof, pvalue)


# In[22]:


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


# In[ ]:





# In[ ]:





# ### 보완

# # Chapter 15: 패널자료 모형(Panel data models)

# In[46]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf 
import statsmodels.api as sm
import statsmodels.stats.api as sms
import scipy.stats as stats 
from statsmodels.iolib.summary2 import summary_col
from statsmodels.graphics import tsaplots 
from statsmodels.tsa.stattools import adfuller 
import seaborn as sns
from scipy.stats import norm
import random
from linearmodels import PooledOLS as pls
from linearmodels import PanelOLS as pnl  
from linearmodels import RandomEffects as pre
from linearmodels import FamaMacBeth as fmb
from linearmodels.panel import FirstDifferenceOLS
from linearmodels.panel import compare 
from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.regression.gmm import IV2SLS as ivs
# Load function from specified path or working directory :
# exec(open("Functions/randefLM.py").read())
# exec(open("Functions/PnlHausman.py").read())


# In[40]:


# Make indexes and print the head of the nls_panel dataset 
nls = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/nls_panel.csv', parse_dates = ['year'], index_col = ['id', 'year']) 
display(nls)

tbl = nls.iloc[0:11, list(range(0, 5)) + list(range(13, 15))] 
display(tbl)

tbl = tbl.reset_index().round(2)
display(tbl)


# In[42]:


# Make indexes and print the head of the nls_panel dataset 
nls = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/nls_panel.csv', 
                  parse_dates = ['year'], index_col = ['id', 'year']) 
display(nls)

tbl = nls.iloc[0:15, [0, 1, 2, 3, 4, 13, 15]]
display(tbl)

tbl = tbl.reset_index().round(2)
display(tbl)


# ## 15.2 패널자료의 회귀분석(Panel data regression)

# In[48]:


# Using PooledOLS on the nls-panel data (from linearmodels import PooledOLS as pis)
formula= """lwage ~ 1 + educ + exper + I(exper**2) + tenure + 
                    I(tenure**2) + black + south + union"""

wg = PooledOLS.from_formula(formula, data = nls)
wgf = wg.fit()
display(wgf.summary) 

dir(wgf)

#wgf.summary
# linear model사용시 formula에 상수항 1을 넣어주어야 gka
# 전체 결가물은 wfg.summary로도 가능


# In[49]:


# 다양한 robust std. errors를 이용한 Pooling OLS  
wgRobust = wg.fit(cov_type = 'robust') 
wgClsId = wg.fit(cov_type = 'clustered', cluster_entity = True) 
wgClsBoth = wg.fit(cov_type = "clustered", cluster_entity = True, cluster_time=True)

# 추정치 테이블 만들기(.compare)
tbl = compare({'unadjusted': wgf, 'robust': wgRobust,
               'clstld': wgClsId, 'clstBoth': wgClsBoth}) 

display(tbl)


# ## 15.3 Fixed effects models
# ### 15.3.2 Two-period production function example

# In[59]:


# OLS 
chem = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical2.csv', parse_dates = ['year'], index_col = ['firm', 'year'])

# 2005, 2006자료
chem56 = chem.loc[:, '2005-01-01':'2006-01-01',:] 
display(chem56)

# Panel OLS
chOLS = PanelOLS.from_formula('lsales ~ 1 + lcapital + llabor', data = chem56) 
chOLSf = chOLS.fit()

display(chOLSf)


# In[60]:


# First Difference OLS 
chFD = FirstDifferenceOLS.from_formula('lsales ~ lcapital + llabor', data = chem56) 
chFDf = chFD.fit()
display(chFDf)


# ### 15.3.4 Within transformation with T = 3

# In[62]:


# Within estimation (EntityEffects를 독립변수로 사용) 
chEE = PanelOLS.from_formula('lsales ~ lcapital + llabor + EntityEffects', data = chem) 
chEEf = chEE.fit()
display(chEEf)


# ### 15.3.6 Fixed effects estimator with T = 3

# In[69]:


# Test for individual (firm) fixed effects 
chem = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical2.csv', 
                   parse_dates = ['year'])


chem['firm'] = pd.Categorical(chem['firm']) 
chLS0 = smf.ols('lsales ~ lcapital + llabor', data = chem).fit()
chLS1 = smf.ols('lsales ~ 1 + lcapital + llabor + firm', data = chem).fit()
anov = anova_lm(chLS0, chLS1)
print('AN0VA comparing OLS vs. individual-dummy models:', '\n', round(anov, 3))


# ## 15.4 Panel data regression error assumptions
# ### 15.4.1 Practice: Pooled OLS with cluster-robust errors

# In[71]:


# Pooled OLS 
chem = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical2.csv', 
                   parse_dates = ['year'], index_col = ['firm', 'year'])

pols = PooledOLS.from_formula('lsales ~ 1 + lcapital + llabor', data = chem) 
polc = pols.fit()
polr = pols.fit(cov_type = 'robust') 
pocl = pols.fit(cov_type = 'clustered', cluster_entity = True)
display(polc)
display(polr)
display(pocl)


# In[75]:


# Driscoll-Kraay standard errors:
ch3 = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical2.csv')

dkls = smf.ols(formula = 'lsales ~ lcapital + llabor', 
               data = ch3).fit(cov_type = 'nw-groupsum', cov_kwds = {'time': np.array(ch3.year),
               'groups': np.array(ch3.firm), 'maxlags': 3})
display(dkls.summary())


# ### 15.4.2 Fixed effects with cluster-robust standard errors

# In[4]:


# Fixed effects with cluster-robust standard errors

ch3 = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical2.csv', 
                   parse_dates = ['year'], index_col = ['firm', 'year'])

chee = PanelOLS.from_formula('lsales ~ 1 + lcapital + llabor + EntityEffects', data=ch3) 

# FE with conventional standard errors
chef = chee.fit() 

# FE with cluster-robust standard errors:
chcr = chee.fit(cov_type = 'clustered', cluster_entity = True) 

tbl1 = pd.concat([chef.params, chef.std_errors, chcr.std_errors], axis = 1)

tbl1.columns = ['coeffs', 'fe.std.er', 'fe.cls-robust'] 
print('Model comparison: FE vs. cluster-robust FE', '\n',
       round(tbl1, 4))


# ## 15.5 Random effects
# ### 15.5.1 Random effects in a production function

# In[5]:


# Random effects
ch3 = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical2.csv', 
                   parse_dates = ['year'], index_col = ['firm', 'year'])

exog_vars = ['lcapital', 'llabor']
exog = sm.add_constant(ch3[exog_vars])

chre =  RandomEffects(ch3.lsales, exog)  
chref = chre.fit()

chrec = chre.fit(cov_type = 'clustered',
        cluster_entity = True, cluster_time = True) 

tbl1 = pd.concat([chref.params, chref.std_errors, chrec.std_errors], axis = 1) 
tbl1.columns = ['coeffs', 're.std.er', 're.cls-robust']
print('Model comparison: RE vs. cluster-robust RE', '\n', round(tbl1, 4))


# In[6]:


# Random effects 
# formula를 사용하는 방법
from linearmodels import RandomEffects  as pre 
ch3 = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical2.csv', 
                   parse_dates = ['year'], index_col = ['firm', 'year'])
chre = RandomEffects.from_formula("lsales ~ 1 + lcapital + llabor", data=ch3)  
chref = chre.fit()

chrec = chre.fit(cov_type = 'clustered',
        cluster_entity = True, cluster_time = True) 

tbl1 = pd.concat([chref.params, chref.std_errors, chrec.std_errors], axis = 1) 

tbl1.columns = ['coeffs', 're.std.er', 're.cls-robust']
print('Model comparison: RE vs. cluster-robust RE', '\n', round(tbl1, 4))


# ### 15.5.2 Random effects in a wage equation

# In[3]:


# Fixed and Random effects 
nls = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/nls_panel.csv', 
                  parse_dates = ['year'], index_col = ['id', 'year']) 

exov = ['educ', 'exper', 'exper2', 'tenure',\
        'tenure2', 'black', 'south', 'union'] 
exog = sm.add_constant(nls[exov]) 

wfe = PanelOLS(nls.lwage, exog, entity_effects = True, drop_absorbed=True).fit() 
wre = RandomEffects(nls.lwage, exog).fit() 

tbl1 = pd.concat([wfe.params, wfe.std_errors, wre.params, wre.std_errors], axis = 1) 
tbl1.columns = [ 'fe.coeffs', 'fe.std.error', 're.coeffs', 're.std.error']
print('Model comparison: FE vs. RE in the "wage" equation',
    '\n', round(tbl1.fillna(''), 4))

# dir(wfe)


# In[4]:


# Fixed and Random effects with the nls_panel dataset 

nls = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/nls_panel.csv', 
                  parse_dates = ['year'], index_col = ['id', 'year']) 

exov = ['educ', 'exper', 'exper2', 'tenure',\
        'tenure2', 'black', 'south', 'union'] 
exog = sm.add_constant(nls[exov]) 

wfe = PanelOLS(nls.lwage, exog, entity_effects = True, drop_absorbed=True).fit() 
wre = RandomEffects(nls.lwage, exog).fit() 

tbl1 = pd.concat([wfe.params, wfe.pvalues, wre.params, wre.pvalues], axis = 1) 
tbl1.columns = [ 'fe.coeffs', 'fe.pvalues', 're.coeffs', 're.pvalues']
print('Model comparison: FE vs. RE in the "wage" equation',
    '\n', round(tbl1.fillna(''), 4))


# In[6]:


# A function to test for random effects. Input: a PooledOLS model.fit() with TimeEffects
def randefLM(pooled_fit):
    e = pooled_fit.resids                            # Retrieve residuals from model fit 
    # Retrieve residual series index sizes:
    T, N = pooled_fit.model.exog.shape[1:3] 
    snum = 0
    sden = 0                                         # Initialize 'entity' sums 
    for i in range(1, N+1) :                         # Loop over entities 
        s1 = 0 
        s2 = 0                                       # Initialize 'time' sums 
        for t in range(0, T):                        # Loop over periods 
            s1 = s1 + e[i][t]                        # sum(e_it) 
            s2 = s2 + e[i][t]**2                     # sum(e_it**2) 
        snum = snum + s1**2                          # Numerator sum 
        sden = sden + s2                             # Denominator sum 
    return np.sqrt(N*T/(2*(T-1)))*(snum/sden - 1)   # LM statistic


# In[7]:


# Random effects test
ch3 = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical3.csv') 
ch3.year = pd.to_datetime(ch3.year, format= '%Y' )

# ch3.year = pd.Categorical (ch3.year) (For your information)
# ch3. dtypes (For your information)
# "drop = False" keeps firm and year as columns:
ch3 = ch3.set_index(['firm', 'year'], drop = False)

# Use PooledOLS to estimate model:
chpl = PanelOLS.from_formula('lsales ~ 1 + lcapital + llabor + TimeEffects', ch3).fit() 
residuals = chpl.resids 
print(chpl)

# Test for random effects:
LM = randefLM(chpl)
LMcr = stats.norm.ppf(1 - 0.01)

print(' Calculated LM = ', round(LM, 4), '\n',
      'Critical LM = ' , round(LMcr, 4), '\n',
      'LM > LMcr => Reject H0')


# ## 15.6 Endogeneity in panel data

# In[14]:


# Hausman test function for panel data. 
# Input: FE.fit() and RE.fit() objects
def PnlHausman(fe_fit, re_fit):
    Dcov = fe_fit.cov - re_fit.cov.iloc[1:, 1:] 
    dparams = fe_fit.params - re_fit.params [1 :]
    Chi2 = dparams.dot(np.linalg.inv(Dcov)).dot(dparams) 
    dof = chref.params.size - 1 
    pvalue = stats.chi2(dof).sf(Chi2) 
    return [Chi2, dof, pvalue]


# In[15]:


# Hausman exogeneity test
ch3 = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical3.csv', 
                  parse_dates = ['year'], index_col = ['firm', 'year'])
# Fixed effects 
chplf = PanelOLS.from_formula('lsales ~ lcapital + llabor + EntityEffects', data = ch3).fit()     
# Random effects
chref = RandomEffects.from_formula('lsales ~ 1 + lcapital + llabor', data = ch3).fit()                           
Chi2, dof, pvalue = PnlHausman(chplf, chref) 
print(Chi2, dof, pvalue)


# In[16]:


# 함수정의하지 않고, Hausman exogeneity test
ch3 = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical3.csv', 
                  parse_dates = ['year'], index_col = ['firm', 'year'])
# Fixed effects 
chplf = PanelOLS.from_formula('lsales ~ lcapital + llabor + EntityEffects', data = ch3).fit()   
# Random effects
chref = RandomEffects.from_formula('lsales ~ 1 + lcapital + llabor', data = ch3).fit()                           

Dcov = chplf.cov - chref.cov.iloc[1:, 1:] 
dparams = chplf.params - chref.params [1 :]

Chi2 = dparams.dot(np.linalg.inv(Dcov)).dot(dparams) 
dof = chref.params.size - 1 
pvalue = stats.chi2(dof).sf(Chi2) 

print(Chi2, dof,pvalue)


# ### 15.6.1 Practice: Hausman test with panel data

# In[17]:


# Practice: Hausman test of endogeneity, production function 
ch3 = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical3.csv', 
                  parse_dates = ['year'], index_col = ['firm', 'year'])
# Fixed effects 
chplf = PanelOLS.from_formula('lsales ~ lcapital + llabor + \
        EntityEffects', data = ch3).fit()  
# Random effects 
chref = RandomEffects.from_formula('lsales ~ 1 + lcapital + llabor', \
        data = ch3).fit()                                 
bfe = chplf.params[0]                         # Fixed effects (lcapital) 
bre = chref.params[1]                         # Random effects (lcapital) 
vbfe = chplf.std_errors[0]**2                 # Fixed effects std.dev. 
vbre = chref.std_errors[1]**2                 #  Random effects std.dev. 
tstat = (bfe - bre)/np.sqrt(vbfe - vbre)      # t-statistic 
dof = chref.nobs - chref.params.size + 1      # Degr. of freedom 
tcr = stats.t.ppf(0.025, dof)                 # Critical 

print(tstat, dof, tcr) 


# ### 15.6.2 Practice: Hausman test in the wage equation

# In[24]:


# Hausman endogeneity test
nls = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/nls_panel.csv', 
                  parse_dates = ['year'], index_col = ['id', 'year']) 

formula1 = """lwage ~ exper + I(exper**2) + tenure + I(tenure**2) + south + union + EntityEffects"""
formula2 = """lwage ~ 1 + exper + I(exper**2) + tenure + I(tenure**2) + south + union"""

fe = PanelOLS.from_formula(formula1, data = nls).fit() 

re = RandomEffects.from_formula(formula2, data = nls).fit()

H, dof, pvalue = PnlHausman(fe, re)

print(H, dof, pvalue) 


# ### 15.6.3 A regression-based Hausman test

# In[27]:


# Mundlak test for three models 
ch3 = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/chemical3.csv', 
                   usecols = ['firm', 'year', 'lsales', 'lcapital', 'llabor']) 
display(ch3a)
 
ch3m = ch3.groupby('firm').mean()                          # Indep. var. means 
display(ch3m)

# 개인별 평균이 원래 데이터세트에 합쳐짐. 이때  _x, _y가 자동으로 붙음
df = pd.merge(ch3, ch3m, on = 'firm', how = 'left')          
display(df)


# In[30]:


formula = 'lsales_x ~ 1 + lcapital_x + llabor_x + lcapital_y + llabor_y'       
hnull = [('lcapital_y = 0.0'), ('llabor_y = 0.0')]  

m1f = smf.ols(formula, df).fit(cov_type='cluster', cov_kwds={'groups': np.array(ch3.firm)})
display(m1f.summary())


# In[33]:


Mstat1 = m1f.wald_test(hnull).statistic 
df = df.set_index(['firm', 'year_x'])                 
print(Mstat1)


# In[34]:


# formula사용해야 되고, statistic가 아니라 stat
# RE, conventional
m2f = RandomEffects.from_formula(formula, df).fit()        
print(m2f)

# Mundlak stat.
Mstat2 = m2f.wald_test(formula = hnull).stat              
print(Mstat2)


# In[36]:


m3f = RandomEffects.from_formula(formula, df).fit(cov_type = 'clustered', 
                                                    cluster_entity = True)
print(m3f)

# Mundlak stat.
Mstat3 = m3f.wald_test(formula = hnull).stat             
print(Mstat3)


# In[37]:


# Show the results of Mundlak statistics with 'chemicals' data
tbl = pd.concat([m1f.params, m1f.bse, m2f.std_errors, m3f.std_errors], axis =1)
tbl.columns = ['coef', 'se.OLScl', 'se.RE', 'se.REcl']
tbl.loc['Mundlak'] = ['', Mstat1, Mstat2, Mstat3]
print(tbl)


# ### 15.6.4 Practice: Mundlak with the wage equation

# In[24]:


wa = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/nls_panel.csv') 
wm = wa.groupby('id').mean()                         
w = wa.merge(wm, on = 'id', how = 'left')            

formula = """lwage_x ~ 1 + educ_x + exper_x + exper2_x + 
             tenure_x + tenure2_x + black_x + south_x + 
             union_x + exper_y + exper2_y + tenure_y +
             tenure2_y + south_y + union_y""" 

hnull = ['exper_y = 0.0', 'exper2_y = 0.0', 'tenure_y = 0.0', 'tenure2_y = 0.0', 'south_y = 0.0', 'union_y = 0.0']
wb = w.set_index(['id', 'year_x'])                   

# RE, conv.
m2f = RandomEffects.from_formula(formula, wb).fit()  

# Mundlak stat. 
Mstat2 = m2f.wald_test(formula = hnull).stat          

m3f = RandomEffects.from_formula(formula, wb).fit(cov_type = 'clustered', 
                    cluster_entity = True)

# Mundlak stat.
Mstat3 = m3f.wald_test(formula = hnull).stat         
print(Mstat3, Mstat3)


# In[25]:


# Fixed-effects model
m4f = PanelOLS.from_formula(formula + ' + EntityEffects', 
                            wb, drop_absorbed = True).fit(cov_type = 'clustered', cluster_entity = True)

# Mundlak statistics 
tbl = pd.concat([m2f.params, m2f.std_errors, m3f.std_errors, m4f.params, m4f.std_errors], axis =1) 

tbl.columns = ['coef.RE', 'se.RE', 'se.RE cluster', 'coef.FE', 'se.FE cluster']
tbl.loc['Mundlak'] = ['', Mstat2, Mstat3, '', ''] 
round(tbl.fillna(''), 4)


# In[42]:


m3f = RandomEffects.from_formula(formula, wb).fit(cov_type = \
    'clustered', cluster_entity = True)
Mstat3 = m3f.wald_test(formula = hnull).stat # Mundlak stat.

# Fixed-effects model
m4f = PanelOLS.from_formula(formula + ' + EntityEffects', wb,
      drop_absorbed = True).fit(cov_type = 'clustered', cluster_entity = True)

# Mundlak statistics 
tbl = pd.concat([m2f.params, m2f.std_errors,
                 m3f.std_errors, m4f.params, m4f.std_errors], axis =1) 
tbl.columns = ['coef.RE', 'se.RE', 'se.REcl', 'coef.FE', 'se.FEcl']
tbl.loc['Mundlak'] = ['', Mstat2, Mstat3, '', ''] 
round(tbl.fillna(''), 4)


# ### 15.6.5 Hausman-Taylor for endogeneous regressor

# In[1]:


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


# In[12]:


# Hausman - Taylor estimation for the 'wage' equation 
df = pd.read_csv('E:/JupyterWorkingDirectory/Principles of Econometrics with Python/Data/nls_panel.csv', 
                 usecols = ['id', 'year', 'lwage', 'educ', 'exper', 'exper2', 'tenure', 'tenure2', 'south', 'black', 'union'],
                 parse_dates = ['year'], index_col = ['id', 'year'])
display(df)

# 기간 수 
T = len(df.index.get_level_values(1).unique())

# Step 1: Fixed effects equation
formula = 'lwage ~ exper + exper2 + tenure + tenure2 + south + union + EntityEffects'
fe = PanelOLS.from_formula(formula, df).fit() 
sige2 = fe.s2 

# 평균값 
dav = df.groupby('id').mean()
dav = pd.merge(df, dav, on = 'id', how = 'left')

# 평균에서의 차이(x - x-bar) 
for name in df.columns:
    dav[name +'_d'] = dav[name + '_x'] - dav[name + '_y']

# 상수항
dav['Intercept'] = np.ones(len(dav))

# Step 2: IV 추정 
iv0 = IV2SLS(dependent = dav.lwage_x,
         exog = dav[['Intercept', 'exper_x', 'exper2_x', 'tenure_x', 'tenure2_x', 'black_x', 'south_x']] ,
         endog = dav['educ_x'],  
         instruments = dav[['Intercept','exper_x', 'exper2_x', 'tenure_x', 'tenure2_x', 'union_x', 'south_x', 'black_x',  
                            'exper_y', 'exper2_y', 'tenure_y', 'tenure2_y', 'union_y']]).fit() 
sigu2 = iv0.scale       



# In[11]:


ahat = 1 - np.sqrt(sige2/(T*sigu2 + sige2))       

# _s 변수 만들기
for name in df.columns:                          
    dav[name +'_s'] = dav[name + '_x'] - ahat * dav[name + '_y']

# Step 3: 최종 IV 모형의 추정 
iv1 = IV2SLS(dependent = dav.lwage_s,
         exog = dav[['Intercept', 'educ_s', 'exper_s', 'exper2_s', 'tenure_s', 
                     'tenure2_s', 'black_s','south_s', 'union_s']],
         instruments = dav[['Intercept', 'exper_s', 'exper2_s', 'tenure_s', 'tenure2_s', 
                           'black_s', 'union_s', 'south_d', 'exper_y', 'exper2_y', 
                           'tenure_y', 'tenure2_y', 'union_y']]).fit() 
display(iv1.summary())

#iv1.summary2().tables[1].iloc[:, :3]     


# In[1]:


# Hausman - Taylor estimation for the 'wage' equation 
df = pd.read_csv(nls_panel.csv', 
                 usecols = ['id', 'year', 'lwage', 'educ', 'exper', 'exper2', 'tenure', 'tenure2', 'south', 'black', 'union'],
                 parse_dates = ['year'], index_col = ['id', 'year'])
display(df)

# 기간 수 
T = len(df.index.get_level_values(1).unique())

# Step 1: Fixed effects equation
formula = 'lwage ~ exper + exper2 + tenure + tenure2 + south + union + EntityEffects'
fe = PanelOLS.from_formula(formula, df).fit() 
sige2 = fe.s2 
df['residual'] = fe.resids


# 평균값 
dav = df.groupby('id').mean()
dav = pd.merge(df, dav, on = 'id', how = 'left')

# 평균에서의 차이(x - x-bar) 
for name in df.columns:
    dav[name +'_d'] = dav[name + '_x'] - dav[name + '_y']

# 상수항
dav['Intercept'] = np.ones(len(dav))
dav['residual'] = fe.resids

# Step 2: IV 추정 
iv0 = ivs(endog = dav.residual,
         exog = dav[['Intercept', 'black_x', 'educ_x']] ,
        instrument = dav[['Intercept','exper_x', 'exper2_x', 'tenure_x', 'tenure2_x', 'union_x', 'black_x']]).fit() 
sig2 = iv0.s2                             
print(sig2)

ahat = 1 - np.sqrt(sige2/(T*sigu2 + sige2))       

# _s 변수 만들기
for name in df.columns:                          
    dav[name +'_s'] = dav[name + '_x'] - ahat * dav[name + '_y']

# Step 3: 최종 IV 모형의 추정 
iv1 = ivs(endog = dav.lwage_s,
         exog = dav[['Intercept', 'educ_s', 'exper_s', 'exper2_s', 'tenure_s', 
                     'tenure2_s', 'black_s','south_s', 'union_s']],
         instrument = dav[['Intercept', 'exper_s', 'exper2_s', 'tenure_s', 'tenure2_s', 'black_s', 'union_s', 
                           'exper_d', 'exper2_d', 'tenure_d', 'tenure2_d', 'union_d', 
                           'black_x']]).fit() 
display(iv1.summary())

#iv1.summary2().tables[1].iloc[:, :3]              


# In[49]:


get_ipython().system('pip install pydynpd')


# In[52]:


from  pydynpd import regression
help(regression)


# In[ ]:


pydynpd.dynamic_panel_model
import pandas as pd
from  pydynpd import regression

df = pd.read_csv("data.csv")
command_str='n L(1:2).n w k  | gmm(n, 2:4) gmm(w, 1:3)  iv(k) | timedumm  nolevel'
mydpd = regression.abond(command_str, df, ['id', 'year'])

