#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# # PART 2: 파이썬 생태계(Python's Ecosystem) 기초

# # 1. 필요한 라이브러리 설치 

# ## 파이썬의 설치
# * 아나콘다(Anaconda) 혹은 윈도우 파이썬(WinPython)
# * https://www.anaconda.com
# * https://www.python.org
# 
# ## 다양한 라이브러리의 설치 또는 갱신(update)
# * 아래 라이브러리 리스트 참조
# * 예) ! pip install package_name
# ## Jupyter Notebook 파일(.ipynb)과 파이썬 파일(.py)의 상호 변환 
# * ! pip install ipynb-py-convert
# * ipynb-py-convert test.ipynb test.py 

# In[4]:


# 연습: 계량경제학 예제로 많이 쓰이는 Wooldridge의 데이터세트 설치
get_ipython().system('pip install wooldridge')


# In[1]:


# 나의 Python Ecosystem에 어떤 Libraries가 설치되어 있는지 확인
conda list


# ## 현재 작업공간(Present Working Directory) 확인 및 바꾸기

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")


# In[2]:


# 현재 작업공간(working directory)확인  
os.getcwd() 


# ## 나만의 작업 디렉터리(Working Directory)를 만들자!
# * D:/My Project/내에 Data, Functions, Figures, Output와 같은 Sub-directory를 만듬
# * 프로젝트 코딩 파일은 D:/My Project/내에 위치시킴
# * Data 디렉터리에는 자료, 
# * Functions 디렉터리에는 사용자가 만든 함수 
# * Figures 디렉터리에는 그래프
# * Output 디렉터리에는 결과물(데이터)을 CSV, Excel, Stata, SAS자료로 출력 보관

# In[3]:


# 메모리 정리
for v in dir():
     del globals()[v]


# ## 필요한 라이브러리(모듈) 불러오기

# In[2]:


get_ipython().system('pip install --upgrade pandas')


# In[ ]:


"""
# 분석에 사용될 라이브러리(Libraries for the Analysis) 불러오기
# Libraries for the Analysis of Traditional Econometrics
# Call this file "exec(open('Functions/Traditional_Econometrics_Lib.py').read())"
import os
import numpy as np                                       # Numerical calculations
import pandas as pd                                      # Data handling
import math as someAlias
import matplotlib.dates as mdates                        # Turn dates into numbers
import matplotlib.pyplot as plt                          # Lower-level graphics
import patsy as pt
import seaborn as sns
import stargazer.stargazer as sg
import statsmodels.api as sm
import statsmodels.formula.api as smf                    # Econometrics
import statsmodels.stats.api as sms
import statsmodels.stats.diagnostic as dg
import statsmodels.stats.outliers_influence as smo
import linearmodels as lm                                # Panel model, Simultaneous Eq. Model
import scipy.stats as stats                              # Statistics
import random

from scipy.optimize import Bounds
from scipy.optimize import curve_fit                    # Nonlinear regression
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.graphics import tsaplots               # Time series
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller          # ADF test
from statsmodels.tsa.stattools import coint             # Cointegration
from statsmodels.tsa.vector_ar.vecm import VECM
from arch import arch_model

import wooldridge as woo
from imfpy.retrievals import dots
import wbdata
import yfinance as yf

import warnings
warnings.filterwarnings("ignore")
"""
# 매 프로젝트 앞에 이를 입력하는 대신 함수(Function)로 정의하여 한줄로 불러올 예정
# 예) exec(open('Functions/Lib_for_Timeseries.py').read()


# In[3]:


exec(open('Functions/Traditional_Econometrics_Lib.py').read())


# In[ ]:


# Spyder
# Jupyter Notebook
# Google Colab


# In[ ]:


# Python, py 파일을 Jupyter Notebook, ipynb 파일로 변환하기 
get_ipython().system(' pip install ipynb-py-convert')
get_ipython().system(' ipynb-py-convert test.py test.ipynb')


# ## 설명문(comment) 입력
# * 샤프(#)로 시작하면 입력된 내용은 파이썬의 실행과 무관한 설명문이 됨.
# * 여러 line을 설명문으로 지정하기 위해서는 """     """ 사용
# * 설명문을 체계적으로 작성하면 차후 이해나 다른 사람과의 공유에 도움

# In[ ]:


# 이하  선형회귀식 추정

""" 
이하 선형회귀 추정과 가설검정
예측을 위한 부분임
"""


# # 2. 간단한 연산 및 출력방법

# In[29]:


# print문의 괄호()내에 수식 기술
print((4 + 6) * 10)


# In[30]:


# 단순히 수식만 입력
100 ** 0.5


# ## 출력(print)방법 3가지 
# ### 1) print문 사용(머신러닝 부분에서 자주 사용)
# * 괄호()내에 f'xxxxxxxxxxxxxxx {var})형식을 이용하여 설명을 위한 텍스트 삽입
# * 빈줄 삽입시 /n 사용
# ### 2) 객체(object)만 사용
# * Jupiter Notebook에서 사용
# ### 3) display문 사용
# * Jupiter Notebook에서 사용
# * 보다 세련된 형식으로 출력 가능

# In[31]:


result1 = 1 + 1
print(f'result1: {result1}\n')


# In[32]:


result1


# In[33]:


display(result1)


# In[34]:


result2 = 5 * (4 - 1) ** 2
print(f'result2: {result2}\n')


# In[35]:


result3 = [result1, result2]
print(f'result3: \n{result3}\n')


# # 3. 모듈의 이해

# In[8]:


import math as someAlias

dir(someAlias)


# In[9]:


result1 = someAlias.log(100)
print(f'result1: {result1}\n')

result2 = someAlias.sqrt(36)
print(f'result1: {result2}\n')

result3 = someAlias.pi
print(f'Pi: {result3}\n')

result4 = someAlias.e
print(f'Eulers number: {result4}\n')


# # 4. 파이썬에서 객체(Objects in Python) 처리

# ## 데이터의 타입

# In[26]:


# 정수(interger)
result1 = 10 + 20
type_result1 = type(result1)
print(f'type_result1: {type_result1}')

# 부동소수(Floating Point Number)
result2 = 25.4
type_result2 = type(result2)
print(f'type_result2: {type_result2}')

# 스트링(String)
result3 = "파이썬에서 객체(Objects in Python)의 이해"
type_result3 = type(result3)
print(f'type_result3: {type_result3}')

# 불린(Boolean)
result4 = True
type_result4 = type(result4)
print(f'type_result4: {type_result4}')

# 리스트(List)
result5 = [10, 15, 20]
type_result5 = type(result5)
print(f'type_result5: {type_result5}')

# 딕셔너리(Dictionary)
result6 = {'A':[10, 20, 30], 'B':[15, 25, 35]}
type_result6 = type(result6)
print(f'type_result6: {type_result6}')


# In[27]:


dir()


# In[28]:


del result1


# In[29]:


# result1이 삭제되었는지 확인?
dir()


# ## 리스트 정의, 요소접근, 치환, 함수, 메써드 적용
# * 인덱스는 0부터 시작
# * 인덱스 선택시 마지막 인텍스는 선택되지 않는다는 점 주의!!!

# In[33]:


# 리스트(ist) 정의
example_list = [10, 20, 100, 40]
print(f'type(example_list): {type(example_list)}')

# 첫번째 index 값 선택
first_entry = example_list[0]
print(f'first_entry: {first_entry}')

# 2번째에서 3번째 index 값 선택
range2to4 = example_list[1:4]
print(f'range2to4: {range2to4}')

# 3번째 Index 값 치환
example_list[2] = 30
print(f'example_list: {example_list}')

# 리스트에 함수 적용(최소값 구하는 함수)
function_output = min(example_list)
print(f'function_min: {function_output}')

# 메써드(method) 적용 
example_list.sort()
print(f'example_list: {example_list}')

# 3번째 Index 값 삭제
del example_list[2]
print(f'example_list: {example_list}')


# ## 딕셔너리의 정의, 변수에의 접근, 치환, 새로운 변수 추가  및 삭제

# In[41]:


# 딕셔너리 정의(1)
var1 = ['Park', 'Kim']
var2 = [170, 190]
var3 = [True, False]
example_dict1 = dict(name=var1, height=var2, diet=var3)
print(f'example_dict1: \n{example_dict1}\n')

# 딕셔너리 정의(2)
example_dict2 = {'name': var1, 'height': var2, 'diet': var3}
print(f'example_dict2: \n{example_dict2}\n')

# 데이터 타입(data type)
print(f'type(example_dict): {type(example_dict1)}\n')

# 특정 변수 출력(여기에서는 'points')
height_all = example_dict1['height']
print(f'height: {height_all}\n')

# Kim의 'height' 출력
height_Kim = example_dict1['height'][1]
print(f'height_Kim : {height_Kim}\n')

# Kim의 height에 10을 더하기 
example_dict1['height'][1] = example_dict1['height'][1] + 5
example_dict1['diet'][1] = True
print(f'example_dict: \n{example_dict1}\n')

# 새로운 변수 weight 추가
example_dict1['weight'] = [70.5, 92.2]
print(f'example_dict: \n{example_dict1}\n')

# 변수 삭제(여기서는 diet)
del example_dict1['diet']
print(f'example_dict: \n{example_dict1}\n')


# # 5. 넘파이(Numpy)에서의 객체
# ## 행렬에서 행(row)과 열(column)의 영문명에 주의!!!
# ## 리스트(list)와 인덱스(index)를 이용한 요소에의 접근시 인덱스의 마지막은 포함되지 않음

# In[46]:


import numpy as np

# 넘파이(numpy)에서 배열(array)의 정의  
array1D = np.array([10, 20, 100, 40])
print(f'type(array1D): {type(array1D)}\n')

array2D = np.array([[10, 30, 50, 70],
                    [20, 40, 60, 80],
                    [30, 60, 90, 40]])

# array2D의 차원(dimention)확인
dim = array2D.shape
print(f'dimention: {dim}\n')

# 배열(array)에서 특정 인덱스(indix) 출력
third_elem = array1D[2]
print(f'third_elem: {third_elem}\n')

# 2번째 행(row)와 3번째 열(column)
second_third_elem = array2D[1, 2]  
print(f'second_third_elem: {second_third_elem}\n')

# 2번째, 3번째 열(column)의 모든 행(row)
second_to_third_col = array2D[:, 1:3]  
print(f'second_to_third_col: \n{second_to_third_col}\n')

# 리스트를 이용한 요소에의 접근
first_third_elem = array1D[[0, 2]]
print(f'first_third_elem: {first_third_elem}\n')

# 불린(Boolean lists)를 이용한 요소에의 접근
first_third_elem2 = array1D[[True, False, True, False]]
print(f'first_third_elem2: {first_third_elem2}\n')

# 불린 배열(Boolean array)를 이용한 요소에의 접근
k = np.array([[True, False, False, False],
              [False, False, True, False],
              [True, False, True, False]])
elem_by_index = array2D[k]  # 1st elem in 1st row, 3rd elem in 2nd row...
print(f'elem_by_index: {elem_by_index}\n')


# ## Numpy의 특수한 1차, 2차원 배열(array)

# In[48]:


import numpy as np

# 시퀀스(수열, Sequence)의 초기, 말기와 길이를 이용하여 배열 생성
sequence = np.linspace(0, 10, num=11)
print(f'sequence: \n{sequence}\n')

# 0부터 10(11-1)까지의 정수 배열 생성
sequence_int = np.arange(11)
print(f'sequence_int: \n{sequence_int}\n')

# 각 원소값이 0인 행열 생성
zero_array = np.zeros((5, 4))
print(f'zero_array: \n{zero_array}\n')

# 각 원소값이 1인 행열 생성
one_array = np.ones((5, 5))
print(f'one_array: \n{one_array}\n')

# 임의의 무의미한 수로 구성된 행열 생성
empty_array = np.empty((5, 4))
print(f'empty_array: \n{empty_array}\n')


# ## Numpy의 2차원 배열(행렬, Matrix)

# In[4]:


import numpy as np

# 넘파이에서 행렬 정의
mat1 = np.array([[24, 19, 38],
                 [28, 62, 39]])
mat2 = np.array([[10, 53, 72],
                 [63, 65, 30],
                 [40, 85, 38]])

# 넘파이에서 함수 사용
result1 = np.log(mat1)
print(f'result1: \n{result1}\n')

result2 = mat1 + mat2[[0, 2]]  
print(f'result2: \n{result2}\n')

result22= np.add(mat1, mat2[[0, 2]])
print(f'result22: \n{result22}\n')

# 넘파이에서 메써드(method) 사용
mat1_trans = mat1.transpose()
print(f'mat1_transpose: \n{mat1_trans}\n')

# 행렬 연산(matrix algebra)
matprod = mat1.dot(mat2) 
print(f'matprod: \n{matprod}\n')

# 동일한 방법의 행렬연산: 
matprod2 = mat1 @ mat2
print(f'matprod2: \n{matprod2}\n')


# # 6. 판다스(Pandas)에서의 객체

# ## 데이터 프레임 만들기, 열(column)과 행(row)의 선택 방법

# In[18]:


import numpy as np
import pandas as pd

# 판다스 데이터 프레임(DataFrame) 정의
sales = np.array([300, 400, 350, 1300, 1200, 600, 900, 800, 1250, 1300, 1350, 1600])
weather = np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
customers = np.array([3000, 3200, 3500, 6000, 7200, 7000, 6000, 7100, 8500, 8200, 8400, 9200])
df = pd.DataFrame({'sales': sales,
                   'weather': weather,
                   'customers': customers})

# 날짜 인덱스 
time_index = pd.date_range(start='01/2025', freq='M', periods=12)
df.set_index(time_index, inplace=True)

# 데이터 프레임의 출력
display(df)
# print(f'df: \n{df}\n')

# 변수명을 이용한 특정 열(columns) 선택
subset1 = df[['sales', 'customers']]
display(subset1)
# print(f'subset1: \n{subset1}\n')

# 특정의 행(row) 선택(2-4행) 
subset2 = df[5:9] 
display(subset2)
# print(f'subset2: \n{subset2}\n')

# 인덱스(index)와 변수명(variable names)으로 행(rows)과 열(columns) 선택: .loc
subset3 = df.loc['2025-06-30', 'customers'] 
display(subset3)
# print(f'subset3: \n{subset3}\n')

# 인덱스(index)와 변수명(variable names)의 위치를 나타내는 정수로 행(rows)과 열(columns) 선택: .iloc
subset4 = df.iloc[5:9, 0:2]
display(subset4)
# print(f'subset4: \n{subset4}\n')


# ## 데이터 프레임에서 시차, 차분, 카테고리, 퍼센트 변화 변수 만들기

# In[11]:


# 시차변수(lag variable) (1, 2기 전)
df['sales_lag1'] = df['sales'].shift(1)
df['sales_lag2'] = df['sales'].shift(2)

# 1계 차분 변수(1st order difference variable
df['sales_diff'] = df['sales'].diff(1)

# 1계 차분 변수(1st order difference variable
df['sales_pch'] = df['sales'].pct_change()

# 카테고리 변수에 레이블(label) 부여 (0 = bad; 1 = good):
df['weather_code'] = pd.Categorical.from_codes(codes=df['weather'],
                                               categories=['나쁨(bad)', '좋음(good)'])

# 원래 데이터 프레임과 새로 추가된 변수 전체 출력
display(df)

# 카테고리별 변수들의 평균값
group_means = df.groupby('weather').mean()
display(group_means)
# print(f'group_means: \n{group_means}\n')


# ## 7. 실제 자료를 이용한 자료 읽고, 저장하기
# ### Wooldridge Dataset "wage1"의 변수명
# #### 자료원: https://www.cengage.com/cgi-wadsworth/course_products_wp.pl?fid=M20b&product_isbn_issn=9781111531041
# #### 24개 변수, 526개 관측치
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

# ## 자료 불러오기(import)

# In[19]:


# 설치된 Wooldridge dataset에서 "wage1" 불러오기
import wooldridge as woo
wage1_sys = woo.dataWoo('wage1')
display(wage1_sys) 

# 다운로드 받은 Wooldridge dataset에서 "wage1.csv" 불러오기
KorAuto = pd.read_csv('data/KorAutoCD.csv')
display(wage1_csv) 

# 다운로드 받은 Wooldridge dataset에서 엑셀 파일 "wage1.xlsx" 불러오기
wage1_excel = pd.read_excel('data/wage1.xlsx', sheet_name='Sheet1' )
display(wage1_excel) 

# 다운로드 받은 Wooldridge dataset에서 Stata 파일 "wage1.dta" 불러오기
wage1_stata = pd.read_stata('data/wage1.dta')
display(wage1_stata) 

# 헤더(Header, 변수명)이 없는 Excel자료 불러오기
wage1_excelnh = pd.read_excel('data/wage1_nh.xlsx', 
                              names = ['wage', 'educ', 'exper', 'tenure', 'nonwhite', 
                                       'female', 'married', 'numdep', 'smsa', 'northcen', 
                                       'south', 'west', 'construc', 'ndurman', 'trcommpu', 
                                       'trade', 'services', 'profserv', 'profocc', 'clerocc', 
                                       'servocc', 'lwage', 'expersq', 'tenursq'])
display(wage1_excelnh) 


# In[23]:


# 데이터 세트의 정보 확인
wage1_excelnh.info()

# csv 파일로 내보내기
wage1_excelnh.to_csv('output/export_sales1.csv')

# Excel파일로 내보내기
wage1_excelnh.to_excel('output/export_sales1.xlsx')

# Stata dataset으로 내보내기
wage1_excelnh.to_stata('output/export_sales1.dta')


# ## 외부 국제기관의 자료 불러오기
# * 세계은행(World Bank): https://wbdata.readthedocs.io/en/stable/
# * 야후(Yahoo Finance) : https://pypi.org/project/yfinance/
# * IMF : https://pypi.org/project/imfpy/

# In[11]:


# WBGAPI설치하고 라이브러리 불러오기
get_ipython().system('pip install wbgapi')
import wbgapi as wb

# World Bank 자료 불러오기 
country = ['KOR', 'USA', 'JPN', 'CHN','GBR', 'FRA', 'DEU', 'ITA']
indicators= ['NY.GDP.MKTP.CD', 'NY.GDP.PCAP.CD', 'NY.GDP.PCAP.PP.CD', 'NY.GDP.DEFL.KD.ZG' 'SP.POP.TOTL', 'EN.ATM.CO2E.KT']
mydata = wb.data.DataFrame(indicators,country, time=range(1990, 2022), skipBlanks=True, columns='series')

# Excel파일로 내보내기
mydata.to_excel('output/mydata.xlsx')

"""
GDP (current US$)
GDP per capita (current US$)
GDP per capita, PPP (current international $)
Inflation, GDP deflator (annual %)
Population, total
CO2 emissions (kt)
"""


# In[2]:


#!pip install yfinance
import yfinance as yf


# In[4]:


# 데이터의 범위 지정  
start = '2010-01-01'  
end   = '2024-01-31'  

# 불러올 주가(코스피, S&P, 나스닥, 니케이)  
assets = ['^KS11', '^GSPC', '^IXIC', '^N225']  
#assets.sort()  

#Downloading price data  
data = yf.download(assets, start=start, end=end)  
data = data.loc[:, ('Adj Close', slice(None))]  
data.columns = assets  
data.columns = ['KOSPI','SNP500', 'NASDAK', 'NIKKEI']
data


# In[14]:


# Excel파일로 내보내기
data.to_excel('output/world_stock.xlsx')


# # 주식 캔들 그래프 그리기

# In[1]:


get_ipython().system('pip install mplfinance')


# In[8]:


import mplfinance as mpf
import yfinance as yf

# Fetch data
aapl_data = yf.download('^KS11', start='2023-06-01', end='2024-03-31')

# Candlestick chart
mpf.plot(aapl_data, type='candle', style='yahoo',
         title='KOSPI Index Chart',
         ylabel='Index',
         volume=True,
         mav=(3,6,9),  # Moving averages
         figratio=(15,6),
         tight_layout=True)


# In[ ]:





# In[26]:


get_ipython().system('pip install imfpy')


# In[30]:


from imfpy.retrievals import dots
dots("KR", ["US", "JP", "CN"], 2000, 2005)


# In[20]:


help(dotsplot)


# In[19]:


from imfpy.tools import dotsplot
dot = dots('KR',['US','CN', 'JP'], 2000, 2020, freq='A', form="long")
dotsplot(dot, subset=['Exports', 'Imports', 'Trade Balance'])


# # 8. 맷플롯립(Matplotlib)을 이용한 그래프 작성

# ### Wooldridge dataset "phillips" 변수명
# #### 7개 변수, 56개 관측치
# * year: 1948 through 2003
# * unem: civilian unemployment rate, percent
# * inf: percentage change in CPI
# * inf_1: inf[_n-1]
# * unem_1: unem[_n-1]
# * cinf: inf - inf_1
# * cunem: unem - unem_1

# In[22]:


import wooldridge as woo
phillips = woo.dataWoo('phillips')
phillips.set_index('year', inplace=True)
display(phillips) 


# ## 기초적인 그래프 그리기

# In[24]:


import matplotlib.pyplot as plt

# create data:
x = phillips['unem']
y = phillips['inf']

plt.plot(x, color='black')
plt.show()

plt.plot(x, color='black', linestyle='--')
plt.show()

plt.plot(x, color='black', linestyle='--', marker='o')
plt.show()

plt.plot(x,y, color='black', linestyle='--', marker='o')
plt.show()


# In[49]:


plt.plot(x, color='black')
plt.plot(y, color='black')
plt.show()


# ## 그래프 저장하기

# In[26]:


plt.plot(x, y, color='black', linestyle=':', marker='o')
plt.show()
plt.savefig('Figures/phillips_curve.pdf'); 
plt.savefig('Figures/phillips_curve.png'); 
plt.close();
# 주의: - 그래프 그리기와 저장은 동시에 해야 함. plt.tight_layout()
#       - plt.show()이후에 저장 명령어를 위치시키면 안됨


# ## 함수, 정규분포 함수 그래프 작성하고, 저장하기

# In[69]:


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# 2차 함수(quadratic function) 정의(-3~3까지 100구간)
x = np.linspace(-3, 3, num=100)

# 이상의 x값에 대해 다음 2차 함수식을 이용한 y값 생성
y = 10+ 2*x + 10*x** 2

# 2차 함수식 도표 그리기 
plt.plot(x, y, linestyle='-', color='black')
plt.tight_layout()
plt.show()


# In[71]:


# 정규분포 밀도함수 작성하기 
x = np.linspace(-4, 4, num=100)
norm_pdf = stats.norm.pdf(x)

# 정규분포의 확률밀도 함수 그리기 
plt.plot(x, norm_pdf, linestyle='-', color='black')
plt.tight_layout()
plt.show()


# ## 다양한 평균, 분산을 가진 정규분포 함수 그리기

# In[82]:


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# 한글폰트 사용하기 위한 모듈
import matplotlib as mpl
mpl.rc('font', family='NanumGothic') # 폰트 설정
mpl.rc('axes', unicode_minus=False) # 유니코드에서 음수 부호 설정

x = np.linspace(-4, 4, num=100)
# 다양한 평균과 분산을 가진 정규분포 밀도함수 계산
y1 = stats.norm.pdf(x, 0, 1)
y2 = stats.norm.pdf(x, 1, 0.5)
y3 = stats.norm.pdf(x, 0, 2)

# 그래그 그리기
# - 그래프 크기와 세부형태 지정
# - Latex 규칙에 의한 수학기호 사용
plt.figure(figsize=(6, 4))
plt.plot(x, y1, linestyle='-', color='black', label='standard normal')
plt.plot(x, y2, linestyle='--', color='0.3', label='$\mu = 1$, $\sigma = 0.5$')
plt.plot(x, y3, linestyle=':', color='0.6', label='$\mu = 0$, $\sigma = 2$')
plt.xlim(-3, 4)

plt.title('정규분포 밀도(Normal Densities) 함수')
plt.ylabel('$\phi(x)$')
plt.xlabel('x')
plt.legend()

# 저장하기
plt.savefig('Figures/Normal_Distribution.png')


# # 주식 캔들 그래프 그리기

# In[1]:


get_ipython().system('pip install mplfinance')


# In[8]:


import mplfinance as mpf
import yfinance as yf

# Fetch data
aapl_data = yf.download('^KS11', start='2023-06-01', end='2024-03-31')

# Candlestick chart
mpf.plot(aapl_data, type='candle', style='yahoo',
         title='KOSPI Index Chart',
         ylabel='Index',
         volume=True,
         mav=(3,6,9),  # Moving averages
         figratio=(15,6),
         tight_layout=True)


# # 9. 실제자료를 이용한 기초통계 및 그래프를 이용한 자료검토
# ### Wooldridge dataset "affairs" 변수명
# #### 19개 변수, 601개 관측치:
# * id: identifier
# * male: =1 if male
# * age: in years
# * yrsmarr: years married
# * kids: =1 if have kids
# * relig: 5 = very relig., 4 = somewhat, 3 = slightly, 2 = not at all, 1 = anti
# * educ: years schooling
# * occup: occupation, reverse Hollingshead scale
# * ratemarr: 5 = vry hap marr, 4 = hap than avg, 3 = avg, 2 = smewht unhap, 1 = vry unhap
# 6 airfare
# * naffairs: number of affairs within last year
# * affair: =1 if had at least one affair
# * vryhap: ratemarr == 5
# * hapavg: ratemarr == 4
# * avgmarr: ratemarr == 3
# * unhap: ratemarr == 2
# * vryrel: relig == 5
# * smerel: relig == 4
# * slghtrel: relig == 3
# * notrel: relig == 2
# 

# In[86]:


import wooldridge as woo
import numpy as np
import pandas as pd

affairs = woo.dataWoo('affairs')

# 결혼만족도(ratemarr)변수의 5개 카테고리, 1~5를 0~4로 조정
affairs['ratemarr'] = affairs['ratemarr'] - 1

# 자식(kids)의 0, 1값을 no, yes값을 가진 자식유무(haskids)변수로 생성
affairs['haskids'] = pd.Categorical.from_codes(affairs['kids'],
                                               categories=['no', 'yes'])
# 결혼 만족도 변수(ratemarr)변수를 이용하여  (0 = 'very unhappy', 1 = 'unhappy',...)로 된 'marriage" 변수 생성 

mlab = ['very unhappy', 'unhappy', 'average', 'happy', 'very happy']
affairs['marriage'] = pd.Categorical.from_codes(affairs['ratemarr'],
                                                categories=mlab)

# 결혼만족도 빈도수 
marrage_freq = np.unique(affairs['marriage'], return_counts=True)
display(marrage_freq)


# In[90]:


# frequency table in pandas:
marrage_freq2 = affairs['marriage'].value_counts()
display(marrage_freq2)


# In[91]:


# 자식유무(haskids)에 따른 결혼만족도(marriage) 빈도수
kid_marr = affairs['marriage'].groupby(affairs['haskids']).value_counts()
display(kid_marr)


# In[94]:


# 교차제표
cr_tab = pd.crosstab(affairs['marriage'], affairs['haskids'], margins=3)
print(f'Cross Table: \n{ct_all_abs}\n')

# 교차제표 전체 구성비
cr_tab_normal = pd.crosstab(affairs['marriage'], affairs['haskids'], normalize='all')
print(f'Cross Table All: \n{cr_tab_normal}\n')

# 교차제표 인덱스별 구성비
cr_tab_row = pd.crosstab(affairs['marriage'], affairs['haskids'], normalize='index')
print(f'Cross Table Row: \n{cr_tab_row}\n')

# 교차제표 변수별 구성비
cr_tab_col = pd.crosstab(affairs['marriage'], affairs['haskids'], normalize='columns')
print(f'Cross Table Col: \n{cr_tab_col}\n')


# In[28]:


import wooldridge as woo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

affairs = woo.dataWoo('affairs')
display(affairs)

# attach labels (see previous script):
affairs['ratemarr'] = affairs['ratemarr'] - 1
affairs['haskids'] = pd.Categorical.from_codes(affairs['kids'],
                                               categories=['no', 'yes'])
mlab = ['very unhappy', 'unhappy', 'average', 'happy', 'very happy']
affairs['marriage'] = pd.Categorical.from_codes(affairs['ratemarr'],
                                                categories=mlab)
display(affairs)


# In[29]:


# 결혼만족도, 자식유무별 결혼만족도에 대한 빈도에 대한 빈도 계산
counts = affairs['marriage'].value_counts()
display(counts)

counts_bykids = affairs['marriage'].groupby(affairs['haskids']).value_counts()
display(counts_bykids)

counts_yes = counts_bykids['yes']
counts_no = counts_bykids['no']
display(counts_yes)
display(counts_no)


# In[37]:


# pie chart (a):
grey_colors = ['0.3', '0.4', '0.5', '0.6', '0.7']
plt.pie(counts, labels=mlab, colors=grey_colors)
plt.tight_layout()
plt.show()


# In[38]:


# horizontal bar chart (b):
y_pos = [0, 1, 2, 3, 4] 
plt.barh(y_pos, counts, color='0.6')
plt.yticks(y_pos, mlab)
plt.tight_layout()
plt.show()


# In[39]:


# stacked bar plot (c)
x_pos = [0, 1, 2, 3, 4] 
plt.bar(x_pos, counts_yes, width=0.4, color='0.6', label='Yes')
# with 'bottom=counts_yes' bars are added on top of previous ones
plt.bar(x_pos, counts_no, width=0.4, bottom=counts_yes, color='0.3', label='No')
plt.ylabel('Counts')
plt.xticks(x_pos, mlab)
plt.legend()
plt.tight_layout()
plt.show()


# In[41]:


# grouped bar plot (d)
# add left bars first and move bars to the left:
plt.figure(figsize=(6, 4))
x_pos_leftbar = [-0.2, 0.8, 1.8, 2.8, 3.8]
plt.bar(x_pos_leftbar, counts_yes, width=0.4, color='0.6', label='Yes')
# add right bars first and move bars to the right:
x_pos_rightbar = [0.2, 1.2, 2.2, 3.2, 4.2]
plt.bar(x_pos_rightbar, counts_no, width=0.4, color='0.3', label='No')
plt.ylabel('Counts')
plt.xticks(x_pos, mlab)
plt.legend()
plt.tight_layout()
plt.show()


# ### Wooldridge dataset "ceosal1" 변수명
# #### 12개 변수 209개 관측치
# * salary: 1990 salary, thousands Dollars
# * pcsalary: percent change salary, 89-90
# * sales: 1990 firm sales, millions  Dollars
# * roe: return on equity, 88-90 avg
# * pcroe: percent change roe, 88-90
# * ros: return on firm’s stock, 88-90
# * indus: =1 if industrial firm
# * finance: =1 if financial firm
# * consprod: =1 if consumer product firm
# * utility: =1 if transport. or utilties
# * lsalary: natural log of salary
# * lsales: natural log of sales
# 

# In[31]:


import wooldridge as woo
import numpy as np

ceosal1 = woo.dataWoo('ceosal1')

# 단순 기술통계량 구하기
ceosal1.describe()
ceosal1.describe().transpose()


# In[30]:


roe = ceosal1['roe']
salary = ceosal1['salary']
consprod = ceosal1['consprod']

# 평균 구하기 
roe_mean = np.mean(salary)
print(f'roe_mean: {roe_mean}\n')

# 중위수 구하기
roe_med = np.median(salary)
print(f'roe_med: {roe_med}\n')

# 표준편차 구하기
roe_s = np.std(salary, ddof=1)
print(f'roe_s: {roe_s}\n')

# roe와 salary의 상관계수 구하기
roe_corr = np.corrcoef(roe, salary)
print(f'roe_corr: \n{roe_corr}\n')


# In[4]:


import wooldridge as woo
import matplotlib.pyplot as plt

ceosal1 = woo.dataWoo('ceosal1')
roe = ceosal1['roe']

# 히스토그램(histogram): 구간 자동구분
plt.hist(roe, color='grey', edgecolor='white')
plt.ylabel('Counts')
plt.xlabel('roe')
plt.show()

# 히스토그램(histogram): 구간 지정
breaks = [0, 5, 10, 20, 30, 60]
plt.hist(roe, color='grey', bins=breaks, density=True, edgecolor='white')
plt.ylabel('density')
plt.xlabel('roe')
plt.show()


# In[5]:


# 커넬 밀도(kernel density)함수 구하기
kde = sm.nonparametric.KDEUnivariate(roe)
kde.fit()
display(kde.density)


# In[6]:


# 커넬밀도(kernel density)함수 그래프
plt.plot(kde.support, kde.density, color='black', linewidth=2)
plt.ylabel('density')
plt.xlabel('roe')
plt.show()


# In[70]:


# 커넬밀도함수와 히스토그램 그리기
plt.hist(roe, color='grey', density=True, edgecolor='white')
plt.plot(kde.support, kde.density, color='black', linewidth=1)
plt.ylabel('density')
plt.xlabel('roe')
plt.show()


# sns.displot(b1, kde=True, bins=20)


# In[11]:


# 박스 그래프 
plt.boxplot(roe)
plt.ylabel('roe')


# In[10]:


# conprod에 따른 roe 박스 그림
roe_0 = roe[consprod == 0]
roe_1 = roe[consprod == 1]

plt.boxplot([roe_0, roe_1])
plt.ylabel('roe')


# # 10. 확률분포(Probability Distribution)

# ## 1) 이산확률분포(discrete distribution)

# In[12]:


import scipy.stats as stats
import math

# 이항분포의 수식을 이용한 확률계산(10개의 공 가운데 4개가 흰색일 때 2개를 뽑을 수 있는 확률)
c = math.factorial(10) / (math.factorial(2) * math.factorial(10 - 2))
p1 = c * (0.4 ** 2) * (0.6 ** 8)
print(f'p1: {p1}\n')

# 사이파이 함수(scipy function)를 이용한 이항분포 pmf계산
p2 = stats.binom.pmf(2, 10, 0.4)
print(f'p2: {p2}\n')


# In[13]:


import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 0~1 사이 11개 균일한 구간의 x값
x = np.linspace(0, 10, num=11)

# x값의 이항 분포의 PMF(probability mass function) 
pmf_binom = stats.binom.pmf(x, 10, 0.4)

# 데이터 프레임 만들기
result = pd.DataFrame({'x': x, 'fx': pmf_binom})
print(f'result: \n{result}\n')

# 이항분호 pmf그리기
plt.bar(x, pmf_binom, color='0.6')
plt.xlabel('x')
plt.ylabel('f(x)')


# ## 2) 연속확률분포(contineous distribution)

# In[14]:


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# 정규분포에서 x값
x = np.linspace(-4, 4, num=100)

# 정규분포의 PDF(probability density function)
norm_pdf = stats.norm.pdf(x)

# 정규분포 그래프 그리기
plt.plot(x, norm_pdf, linestyle='-', color='black')
plt.xlabel('x')
plt.show()


# In[15]:


import scipy.stats as stats

# 이항분포의 누적밀도 함수(CDF)
p1 = stats.binom.cdf(2, 10, 0.4)
print(f'p1: {p1}\n')

# 정규분포의 누적밀도 합수(CDF)
p2 = stats.norm.cdf(1.96) - stats.norm.cdf(-1.96)
print(f'p2: {p2}\n')


# In[16]:


import scipy.stats as stats

# 변수 X가 평균 4, 분산 9인 정규분포, 즉 X~n(4, 9)할 때 
# P(2<X<6) = q(2/3)-q(-2/3)이 됨. 그 면적을 구하려면
norm_21 = stats.norm.cdf(2 / 3) - stats.norm.cdf(-2 / 3)
print(f'norm_21: {norm_21}\n')

# 이상의 사례는 다음과 동일함
norm_22 = stats.norm.cdf(6, 4, 3) - stats.norm.cdf(2, 4, 3)
print(f'norm_22: {norm_22}\n')

# 평균 4, 표준편차 3인 정규분포에서 P(|X|>2) = 1 - P(X<=2) + P(X<=-2)의 확률은? 
norm_3 = 1 - stats.norm.cdf(2, 4, 3) + stats.norm.cdf(-2, 4, 3)
print(f'norm_3: {norm_3}\n')


# In[6]:


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# 이항분포(binomial distribution)
x_binom = np.linspace(-1, 10, num=1000)

# 이항분포의 PMF 
cdf_binom = stats.binom.cdf(x_binom, 10, 0.4)

# 이항분포 CDF의 그래프
plt.step(x_binom, cdf_binom, linestyle='-', color='black')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.show()

# 정규분포(normaldistribution)의 X범위
x_norm = np.linspace(-4, 4, num=1000)

# 정규분포의 PDF 
cdf_norm = stats.norm.cdf(x_norm)

# 로지스틱 분포의 PDF 
cdf_logt = stats.logistic.cdf(x_norm)

# 정규분포, 로지스틱 분포의 CDF 그래프
plt.plot(x_norm, cdf_norm, linestyle='-', color='black')
plt.plot(x_norm, cdf_logt, linestyle='--', color='black')
plt.xlabel('x')
plt.ylabel('Fx')
plt.show()


# In[18]:


# 분위수 함수(quantile function): .ppf
import scipy.stats as stats

q_975 = stats.norm.ppf(0.975)
print(f'q_975: {q_975}\n')


# ## 3) 확률분포의 난수만들기(random draw)

# In[8]:


import numpy as np
import scipy.stats as stats

ber = stats.bernoulli.rvs(0.5, size=10)
nor = stats.norm.rvs(size=10)
print(ber, nor)

"""
ber = stats.bernoulli.rvs(0.5, size=10)
ber = stats.binom.rvs(0.5, size=10)
uni = stats.uniform.rvs()
nor = stats.norm.rvs()
bet = stats.beta.rvs()
gam = stats.gamma.rvs()
ttt = stats.t.rvs()
chi = stats.chi2.rvs(50, size=10)
fff = stats.F.rvs(50, 20, size=10)
mnorr = stats.multivariate_normal.rvs()
rand =pd.DataFrame
display(rand)
print(ber)
"""


# In[19]:


# 표준정규분포(standard normal) RV (size n=10:
sample1 = stats.norm.rvs(size=10)
print(f'sample1: {sample1}\n')

sample2 = stats.norm.rvs(size=10)
print(f'sample2: {sample2}\n')

# random number 만들기 위한 seed(매 실행시 동일한 결과)
np.random.seed(123456)
sample3 = stats.norm.rvs(size=10)
print(f'sample3: {sample3}\n')


# In[9]:


import numpy as np
import scipy.stats as stats

# 2변수 차이의 평균의 신뢰구간
SR87 = np.array([10, 1, 6, .45, 1.25, 1.3, 1.06, 3, 8.18, 1.67,
                 .98, 1, .45, 5.03, 8, 9, 18, .28, 7, 3.97])

SR88 = np.array([3, 1, 5, .5, 1.54, 1.5, .8, 2, .67, 1.17, .51,
                 .5, .61, 6.7, 4, 7, 19, .2, 5, 3.83])

# 변화
Change = SR88 - SR87

# 평균과 그 표준편차
avgCh = np.mean(Change)

n = len(Change)
sdCh = np.std(Change, ddof=1)
se = sdCh / np.sqrt(n)

c = stats.t.ppf(0.975, n - 1)

# 신뢰구간(confidence interval)
lowerCI = avgCh - c * se
upperCI = avgCh + c * se
print(f'upperCI: {upperCI}\n')
print(f'lowerCI: {lowerCI}\n')


# ## 11. 신뢰구간과 통계적 추론
# #### 3개변수, 241개 관측치 
# * w: =1 if white app. got job offer
# * b: =1 if black app. got job offer
# * y: b - w

# In[20]:


import wooldridge as woo
import numpy as np
import scipy.stats as stats

audit = woo.dataWoo('audit')
y = audit['y']

# 신뢰구간 계산
avgy = np.mean(y)
n = len(y)
sdy = np.std(y, ddof=1)
se = sdy / np.sqrt(n)
c95 = stats.norm.ppf(0.975)
c99 = stats.norm.ppf(0.995)

# 95% confidence interval:
lowerCI95 = avgy - c95 * se
upperCI95 = avgy + c95 * se
print(f'lowerCI95: {lowerCI95}\n')
print(f'upperCI95: {upperCI95}\n')

# 99% confidence interval:
lowerCI99 = avgy - c99 * se
upperCI99 = avgy + c99 * se
print(f'lowerCI99: {lowerCI99}\n')
print(f'upperCI99: {upperCI99}\n')


# In[21]:


import numpy as np
import pandas as pd
import scipy.stats as stats

# 자유도 = n-1:
df = 19

# 유의수준
alpha_one_tailed = np.array([0.1, 0.05, 0.025, 0.01, 0.005, .001])
alpha_two_tailed = alpha_one_tailed * 2

# 임계치 
CV = stats.t.ppf(1 - alpha_one_tailed, df)
table = pd.DataFrame({'alpha_one_tailed': alpha_one_tailed,
                      'alpha_two_tailed': alpha_two_tailed, 
                      'CV': CV})
print(f'table: \n{table}\n')


# In[22]:


import wooldridge as woo
import numpy as np
import pandas as pd
import scipy.stats as stats

audit = woo.dataWoo('audit')
y = audit['y']

# t-통계를 이용한 평균 0의 검정(함수이용)
test_auto = stats.ttest_1samp(y, popmean=0)
t_auto = test_auto.statistic  
p_auto = test_auto.pvalue  
print(f't_auto: {t_auto}\n')
print(f'p_auto/2: {p_auto / 2}\n')

# t-통계를 이용한 평균 0의 검정(수식이용)
avgy = np.mean(y)
n = len(y)
sdy = np.std(y, ddof=1)
se = sdy / np.sqrt(n)
t_manual = avgy / se
print(f't_manual: {t_manual}\n')

# t-통계의 유의수준별 임계값(critical values)  
alpha_one_tailed = np.array([0.1, 0.05, 0.025, 0.01, 0.005, .001])
CV = stats.t.ppf(1 - alpha_one_tailed, 240)
table = pd.DataFrame({'alpha_one_tailed': alpha_one_tailed, 
                      'CV': CV})
print(f'table: \n{table}\n')


# In[24]:


import numpy as np
import scipy.stats as stats

SR88 = np.array([10, 1, 6, .45, 1.25, 1.3, 1.06, 3, 8.18, 1.67,
                 .98, 1, .45, 5.03, 8, 9, 18, .28, 7, 3.97])
SR87 = np.array([3, 1, 5, .5, 1.54, 1.5, .8, 2, .67, 1.17, .51,
                 .5, .61, 6.7, 4, 7, 19, .2, 5, 3.83])
Change = SR88 - SR87

# t-통계를 이용한 평균 0의 검정(함수이용)
test_auto = stats.ttest_1samp(Change, popmean=0)
t_auto = test_auto.statistic
p_auto = test_auto.pvalue
print(f't_auto: {t_auto}\n')
print(f'p_auto/2: {p_auto / 2}\n')

# t-통계를 이용한 평균 0의 검정(수식이용)
avgCh = np.mean(Change)
n = len(Change)
sdCh = np.std(Change, ddof=1)
se = sdCh / np.sqrt(n)
t_manual = avgCh / se
print(f't_manual: {t_manual}\n')

# p value
p_manual =1- stats.t.cdf(t_manual, n - 1)
print(f'p_manual: {p_manual}\n')


# In[3]:


import wooldridge as woo
import numpy as np
import pandas as pd
import scipy.stats as stats

# 흑인과 백인의 고용율 차이(y) 여부 검정
audit = woo.dataWoo('audit')
y = audit['y']

# 1) t-통계 수식을 이용한 평균 0(차이없음)의 검정
avgy = np.mean(y)
n = len(y)
sdy = np.std(y, ddof=1)
se = sdy / np.sqrt(n)
t_manual = avgy / se
print(f't_manual: {t_manual}\n')

# 2) t-통계함수를 이용한 평균 0(차이없음)의 검정
test_auto = stats.ttest_1samp(y, popmean=0)
t_auto = test_auto.statistic  
p_auto = test_auto.pvalue  
print(f't_auto: {t_auto}\n')
print(f'p_auto/2: {p_auto / 2}\n')

# 3) t-통계의 유의수준별 임계값(critical values)  
alpha_one_tailed = np.array([0.05, 0.025, 0.01, 0.005])
CV = stats.t.ppf(1 - alpha_one_tailed, 240)
table = pd.DataFrame({'alpha_one_tailed': alpha_one_tailed, 
                      'CV': CV})
print(f'table: \n{table}\n')  

# 4) t-통계의 p-value 계산
p_manual = stats.t.cdf(t_manual, n-1)
print(f'p_manual: {p_manual} \n')

# 5) 신뢰구간 계산
c95 = stats.norm.ppf(0.975)
c99 = stats.norm.ppf(0.995)

# - 95% confidence interval:
lowerCI95 = avgy - c95 * se
upperCI95 = avgy + c95 * se
print(f'lowerCI95: {lowerCI95}\n')
print(f'upperCI95: {upperCI95}\n')

# - 99% confidence interval:
lowerCI99 = avgy - c99 * se
upperCI99 = avgy + c99 * se
print(f'lowerCI99: {lowerCI99}\n')
print(f'upperCI99: {upperCI99}\n')



# ## 12. Python 보완

# In[45]:


# 1.8 Advanced Python


seq = [1, 2, 3, 4, 5, 6]
for i in seq:
    if i < 4:
        print(i ** 3)
    else:
        print(i ** 2)


# In[46]:


seq = [10, 20, 30, 40, 50, 60]
for i in range(len(seq)):
    if seq[i] < 40:
        print(seq[i] ** 2)
    else:
        print(seq[i] ** 1)


# In[47]:


# 함수(function)의 정의
def root(x):
    if x >= 0:
        result = x ** 0.5
    else:
        result = 'Not defined'
    return result

# 함수호출하여 계산결과 보관
result1 = root(100)
print(f'result1: {result1}\n')

result2 = root(-100)
print(f'result2: {result2}\n')


# In[53]:


# use the predefined class 'list' to create an object:
a = [1, 5, 2, 4, 5]

# access a local variable (to find out what kind of object we are dealing with):
dir(type(a))


# In[52]:


check = type(a).__name__
print(f'check: {check}\n')

# make use of a method (how many 5 are in a?):
count_5 = a.count(5)
print(f'count_5: {count_5}\n')

# use another method (sort data in a):
a.sort()
print(f'a: {a}\n')


# In[128]:


import numpy as np

# multiply these two matrices:
a = np.array([[3, 6, 1], [2, 7, 4]])
b = np.array([[1, 8, 6], [3, 5, 8], [1, 1, 2]])

# the numpy way:
result_np = a.dot(b)
print(f'result_np: \n{result_np}\n')

# or, do it yourself by defining a class:
class myMatrices:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def mult(self):
        N = self.A.shape[0]  # number of rows in A
        K = self.B.shape[1]  # number of cols in B
        out = np.empty((N, K))  # initialize output
        for i in range(N):
            for j in range(K):
                out[i, j] = sum(self.A[i, :] * self.B[:, j])
        return out


# create an object:
test = myMatrices(a, b)

# access local variables:
print(f'test.A: \n{test.A}\n')
print(f'test.B: \n{test.B}\n')

# use object method:
result_own = test.mult()
print(f'result_own: \n{result_own}\n')


# In[129]:


import numpy as np

# multiply these two matrices:
a = np.array([[3, 6, 1], [2, 7, 4]])
b = np.array([[1, 8, 6], [3, 5, 8], [1, 1, 2]])


# define your own class:
class myMatrices:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def mult(self):
        N = self.A.shape[0]  # number of rows in A
        K = self.B.shape[1]  # number of cols in B
        out = np.empty((N, K))  # initialize output
        for i in range(N):
            for j in range(K):
                out[i, j] = sum(self.A[i, :] * self.B[:, j])
        return out


# define a subclass:
class myMatNew(myMatrices):
    def getTotalElem(self):
        N = self.A.shape[0]  # number of rows in A
        K = self.B.shape[1]  # number of cols in B
        return N * K


# create an object of the subclass:
test = myMatNew(a, b)

# use a method of myMatrices:
result_own = test.mult()
print(f'result_own: \n{result_own}\n')

# use a method of myMatNew:
totalElem = test.getTotalElem()
print(f'totalElem: {totalElem}\n')


# ## 13. 몬테칼로 시뮬레이션(Monte Carlo Simulation)

# In[56]:


import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# random seed:
np.random.seed(123456)

# 표본수:
n = 10000

# 반복횟수 지정, 평균값 보관할 빈공간:
r = 10000
ybar = np.empty(r)

# 반복 10000회:
for j in range(r):
    # 반복하면서 정규분포 샘플생성후 평균값 계산:
    sample = stats.norm.rvs(10, 2, size=n)
    ybar[j] = np.mean(sample)
print(ybar) 
print(ybar.mean()) 
print(ybar.std()) 

# 커넬 밀도함수
kde = sm.nonparametric.KDEUnivariate(ybar)
kde.fit()

# 정규분포 밀도함수(normal density):
x_range = np.linspace(9, 11, num=100)
y = stats.norm.pdf(x_range, 10, np.sqrt(4/n))
print(y)

# 그래프 그리기:
plt.plot(kde.support, kde.density, color='black', label='ybar')
plt.plot(x_range, y, linestyle='--', color='black', label='Normal distribution')
plt.ylabel('density')
plt.xlabel('ybar')
plt.legend()


# In[52]:


import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# random seed:
np.random.seed(123456)

# 표본수:
n = 1000

# 반복횟수 지정, 평균값 보관할 빈공간:
r = 10000
ybar = np.empty(r)

# 반복 10000회:
for j in range(r):
    # 반복하면서 Ch2 샘플생성후 평균값 계산:
    sample = stats.chi2.rvs(1, size=n)
    ybar[j] = np.mean(sample)
print(ybar) 
print(ybar.mean()) 
print(ybar.std()) 

# 커넬 밀도함수
kde = sm.nonparametric.KDEUnivariate(ybar)
kde.fit()

# 정규분포 밀도함수(normal density):
x_range = np.linspace(0.5, 1.5, num=100)
y = stats.norm.pdf(x_range, 1, np.sqrt(4/n))
print(y)

# 그래프 그리기:
plt.plot(kde.support, kde.density, color='black', label='ybar')
plt.plot(x_range, y, linestyle='--', color='black', label='Normal distribution')
plt.ylabel('density')
plt.xlabel('ybar')
plt.legend()


# In[20]:


import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# random seed:
np.random.seed(123456)

# 표본수 지정:
n = 100

# # 반복횟수 지정, 평균값 보관할 빈공간::
r = 10000
ybar = np.empty(r)

# 반복 10000회:
for j in range(r):
    sample = stats.chi2.rvs(1, size=n)
    ybar[j] = np.mean(sample)
print(ybar) 

# 커넬 밀도함수
kde = sm.nonparametric.KDEUnivariate(ybar)
kde.fit()

# 정규분포 밀도함수(Normal density):
x_range = np.linspace(0, 4, num=100)
y = stats.chi2.pdf(x_range, 1, np.sqrt(4/n))
print(y)

# 그래프 그리기:
plt.plot(kde.support, kde.density, color='black', label='ybar')
plt.plot(x_range, y, linestyle='--', color='black', label='Normal distribution')
plt.ylabel('density')
plt.xlabel('ybar')
plt.legend()


# In[30]:


import numpy as np
import scipy.stats as stats

np.random.seed(123456)

# 표본수와 반복수
r = 10000
n = 1000

# p-value 보관을 위한 빈공간
pvalue1 = np.empty(r)
pvalue2 = np.empty(r)

for j in range(r):
    sample = stats.norm.rvs(10, 2, size=n)
    sample_mean = np.mean(sample)
    sample_sd = np.std(sample, ddof=1)

    # 평균=10 테스트 
    testres1 = stats.ttest_1samp(sample, popmean=10)
    pvalue1[j] = testres1.pvalue

    # 평균=9.5 테스트 
    testres2 = stats.ttest_1samp(sample, popmean=9.5)
    pvalue2[j] = testres2.pvalue

# 검정결과(기각하는 것이 참(True):
# 평균=10을 기각
reject1 = pvalue1 <= 0.05
count1_true = np.count_nonzero(reject1) 
count1_false = r - count1_true
print(f'count1_true: {count1_true}\n')
print(f'count1_false: {count1_false}\n')

# 평균=9.5를 기각
reject2 = pvalue2 <= 0.05
count2_true = np.count_nonzero(reject2)
count2_false = r - count2_true
print(f'count2_true: {count2_true}\n')
print(f'count2_false: {count2_false}\n')    


# In[31]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# random seed:
np.random.seed(123456)

# 표본수와 반복수:
r = 10000
n = 1000

# 신뢰구간 보관을 위한 빈공간:
CIlower = np.empty(r)
CIupper = np.empty(r)

# r회 반복:
for j in range(r):
    sample = stats.norm.rvs(10, 2, size=n)
    sample_mean = np.mean(sample)
    sample_sd = np.std(sample, ddof=1)

    # 평균의 신뢰구간 계산
    cv = stats.t.ppf(0.975, df=n - 1)
    CIlower[j] = sample_mean - cv * sample_sd / np.sqrt(n)
    CIupper[j] = sample_mean + cv * sample_sd / np.sqrt(n)

# 평균=10 포함 신뢰구간(처음 200회 반복결과 이용)
plt.figure(figsize=(3, 5)) 
plt.ylim(0, 201)
plt.xlim(9, 11)
for j in range(1, 201):
    if 10 > CIlower[j] and 10 < CIupper[j]:
        plt.plot([CIlower[j], CIupper[j]], [j, j], linestyle='-', color='grey')
    else:
        plt.plot([CIlower[j], CIupper[j]], [j, j], linestyle='-', color='black')
plt.axvline(10, linestyle='--', color='black', linewidth=0.5)
plt.ylabel('Sample No.')


# In[32]:


# 평균=9.5 포함 신뢰구간(처음 200회 반복결과 이용)
plt.figure(figsize=(3, 5))  
plt.ylim(0, 201)
plt.xlim(9, 11)

for j in range(1, 201):
    if 9.5 > CIlower[j] and 9.5 < CIupper[j]:
        plt.plot([CIlower[j], CIupper[j]], [j, j], linestyle='-', color='grey')
    else:
        plt.plot([CIlower[j], CIupper[j]], [j, j], linestyle='-', color='black')
plt.axvline(9.5, linestyle='--', color='black', linewidth=0.5)
plt.ylabel('Sample No.')


# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


uni = stats.uniform.rvs(size=10000, loc = 10, scale=20)
exp = stats.expon.rvs(scale=1,loc=0,size=10000) 
bin = stats.binom.rvs(n=10,p=0.8,size=10000) 
poi = stats.poisson.rvs(mu=3, size=10000) 
ber = stats.bernoulli.rvs(size=10000,p=0.6) 
geo = stats.geom.rvs(p=0.5, size=10000)
wei = stats.weibull_min.rvs(c=1.79, size=10000)
nor = stats.norm.rvs(size=10000,loc=10,scale=20)  
snor = stats.norm.rvs(size=10000,loc=0,scale=1)
lnor = stats.lognorm.rvs(s=0.954, size=10000)
t = stats.t.rvs(df=35, size=10000)
f = stats.f.rvs(dfn=29, dfd=18, size=10000)
ch2 = stats.chi2.rvs(df=5, size=10000) 
gam = stats.gamma.rvs(a=3, size=10000) 
bet = stats.beta.rvs(a=2.3, b=0.6, size=10000) 
logi = stats.logistic.rvs(size=10000)


# In[9]:


logi


# In[ ]:


df = pd.DataFrame(np.vstack((uni, exp, bin, poi, ber, geo, wei, nor, snor, lnor, t, f, ch2, gam, bet, logi))).T

col = "uni exp bin poi ber geo wei nor snor lnor t f ch2 gam bet logi"
columns = col.split() 
df.columns = columns
display(df)


# In[20]:


df.plot.hist(subplots = True, layout = (4,4), sharex = False, bins = 40, colormap="gist_rainbow");


# In[22]:


df['snor'].plot.hist(bins = 40, colormap="gist_rainbow");


# In[26]:


# subplots으로 그리기
figure, ax = plt.subplots(4, 4, figsize=(12,8))
sns.histplot(uni,  kde=True, ax=ax[0,0]).set(title='Uniform')
sns.histplot(exp,  kde=True, ax=ax[0,1]).set(title='Exponential')
sns.histplot(bin,  kde=True, ax=ax[0,2]).set(title='Binomial')
sns.histplot(poi,  kde=True, ax=ax[0,3]).set(title='Poisson') 
sns.histplot(ber,  kde=True, ax=ax[1,0]).set(title='Bernoulli')
sns.histplot(geo,  kde=True, ax=ax[1,1]).set(title='Geometric') 
sns.histplot(wei,  kde=True, ax=ax[1,2]).set(title='Weibull') 
sns.histplot(nor,  kde=True, ax=ax[1,3]).set(title='Normal') 
sns.histplot(snor, kde=True, ax=ax[2,0]).set(title='Standard Normal')
sns.histplot(lnor, kde=True, ax=ax[2,1]).set(title='Log Normal') 
sns.histplot(t,    kde=True, ax=ax[2,2]).set(title='t') 
sns.histplot(f,    kde=True, ax=ax[2,3]).set(title='F') 
sns.histplot(ch2,  kde=True, ax=ax[3,0]).set(title='Chi-Squared')
sns.histplot(gam,  kde=True, ax=ax[3,1]).set(title='Gamma') 
sns.histplot(bet,  kde=True, ax=ax[3,2]).set(title='Beta') 
sns.histplot(logi,  kde=True, ax=ax[3,3]).set(title='Logistic') 
plt.tight_layout()
plt.show()


# In[20]:


# Uniform distribution
rv_array = spss.uniform.rvs(size=10000, loc = 10, scale=20)

sns.histplot(rv_array, kde=True)   # plotted using seaborn


# In[4]:


# plotted using seaborn
rv_df = pd.DataFrame(rv_array, columns=['value_of_random_variable'])
sns.histplot(data=rv_df, x='value_of_random_variable', kde=True)    


# In[ ]:


# Normal Distribution


# In[5]:


rv_array = spss.norm.rvs(size=10000,loc=10,scale=100)  
sns.histplot(rv_array, kde=True) 


# In[6]:


ax = sns.distplot(rv_array, bins=100, kde=True, color='cornflowerblue', hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Normal Distribution', ylabel='Frequency')


# In[ ]:


# Standard Normal Distribution


# In[7]:


rv_array = spss.norm.rvs(size=10000,loc=0,scale=1) 
sns.histplot(rv_array, kde=True)


# In[ ]:


# Gamma distribution 


# In[8]:


rv_array = spss.gamma.rvs(a=5, size=10000) 
sns.displot(rv_array, kde=True)


# In[ ]:


# Exponential distribution


# In[10]:


rv_array = spss.expon.rvs(scale=1,loc=0,size=1000) 
sns.displot(rv_array, kde=True)


# In[ ]:


# Binomial Distribution


# In[11]:


rv_array = spss.binom.rvs(n=10,p=0.8,size=10000) 
sns.displot(rv_array, kde=True)


# In[ ]:


# Poisson Distribution


# In[13]:


rv_array = spss.poisson.rvs(mu=3, size=10000) 
sns.displot(rv_array, kde=True)


# In[ ]:


# Bernoulli distribution


# In[14]:


rv_array = spss.bernoulli.rvs(size=10000,p=0.6) 
sns.displot(rv_array, kde=True)


# In[ ]:




