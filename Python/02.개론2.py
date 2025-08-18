#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# # 3. 데이터 프래임의 관리와 변수 처리

# ## 3.1	데이터 프래임(DataFrames)의 관리(Manipulation)

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[2]:


exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[3]:


get_ipython().system('pip install --upgrade pandas')


# In[5]:


import numpy as np
import pandas as pd

# 외부 데이터 불러오기
data = pd.read_csv('Data/insurance.csv')
display(data.head())


# In[9]:


# 변수의 정보(관측치수, 데이터 형태) 확인
data.info()


# In[7]:


# 정량적 변수(quantitative variables)의 기술 통계량 계산(discriptive statistics)
data.describe().transpose()


# In[8]:


# 정성적 변수(qualitative variables)의 카테고리(categories)확인 1
display(data['sex'].unique()) 
display(data['smoker'].unique()) 
display(data['region'].unique())


# In[14]:


# 정성적 변수(qualitative variables)의 카테고리(categories)확인 2
for i in data.columns.tolist(): 
        if data[i].dtype == 'O':
            display(data[i].unique().tolist())
# data의 column이름을 리스트로 해서, dtype을 확인한 후 
# object, 즉 0이면 변수명에 대해 unique한 값을 리스트로 출력


# In[10]:


# 변수명 바꾸기
df = data.copy()
newcols = {'age': '나이', 
           'sex' : '성별',
           'bmi' : '체질량지수',
           'children' : '어린이 수',
           'smoker': '흡연여부', 
           'region': '거주지역',
           'charges': '보험액'}
df.rename(columns=newcols, inplace=True)
display(df.head())


# In[23]:


# 정성적 변수의 카테고리별 빈도수
display(data['sex'].value_counts())
display(data['smoker'].value_counts())
display(data['region'].value_counts())


# In[11]:


# 통계함수(Statistical functions) 사용예 1
print('sum:', data['charges'].sum())  
print('count:', data['charges'].count())  
print('max:', data['charges'].max())  
print('min:', data['charges'].min())  
print('mean:', data['charges'].mean())  
print('median:', data['charges'].median())  
print('mode:', data['charges'].mode())  
print('std:', data['charges'].std())  
print('var:', data['charges'].var())  
print('skew:', data['charges'].skew())  
print('kurt:', data['charges'].kurt()) 


# In[25]:


# 통계함수(Statistical functions) 사용예 2 
stats = ['sum', 'count', 'max', 'min', 'mean', 'median',  'mode', 'std', 'var', 'skew', 'kurt']  

for i in stats:  
    stats = getattr(data['charges'], i)()  
    print(i + ' : ', stats) 


# In[34]:


# 통계함수(Statistical functions) 사용예 2 
vars =['age', 'bmi', 'children', 'charges']
stat = ['sum', 'count', 'max', 'min', 'mean', 'median',  'mode', 'std', 'var', 'skew', 'kurt']  

for i in vars:  
    for j in stat:  
        stats = getattr(data[i], j)()  
        print(i + ' : ' + j + ' : ', stats) 


# In[37]:


# 0.9 분위수 계산
display(data.quantile(0.9))


# In[38]:


# 카테고리별 변수, 지역별 숫자변수의 평균
data.groupby('region').mean() 


# In[39]:


# 여러 개 카테고리별 숫자변수의 평균  
table = data.groupby(['sex', 'region']).mean().applymap('{:,.2f}'.format)  
display(table) 


# In[46]:


# 데이터 프레임(Data Frame)의 결합(Append and Merge)
data_1 = {'id': [1, 2, 3, 4, 5],  
          'first': ['AA', 'BB', 'CC', 'DD', 'EE'],  'last': ['FF', 'GG', 'HH', 'II', 'JJ']}  
data_2 = {'id': [3, 4, 5, 6, 7],  'Sales': [100, 200, 300, 400, 500]}  

data_1 = pd.DataFrame(data_1)  
data_2 = pd.DataFrame(data_2)  

# 겹치는 ID만 결합
inner_join = pd.merge(data_1, data_2, on = 'id')  
print(inner_join)  

# 모든 ID 결합
outer_join = pd.merge(data_1, data_2, on = 'id', how = 'outer')  
print(outer_join)  

# 왼쪽 데이터 프레임 기준
left_join = pd.merge(data_1, data_2, on = 'id', how = 'left')  
print(left_join)  

# 오른쪽 데이터 프레임 기준
right_join = pd.merge(data_1, data_2, on = 'id', how = 'right')  
print(right_join) 


# In[56]:


# Wide form --> long form : pd.wide_to_long
data = pd.read_csv('Data/wideform.csv')
print(data.head())

df = pd.wide_to_long(data, ["A", "B"], i="id", j="year")
df.sort_values(by=['id', 'year'], inplace=True)
print(df)

desc1 = df.groupby(level=0, axis=0).describe()
desc2 = df.groupby(level=1, axis=0).describe()

# ID별 평균에 대한 편차
df_mean = df.groupby(level=0, axis=0).transform('mean')
df_mean.columns = ['A_mean', 'B_mean']
print(df_mean)

# 왼쪽 데이터 프레임 기준
df_join = pd.merge(df, df_mean, on = ['id', 'year'], how = 'left')  
print(df_join)  


# In[43]:


df = df.reset_index()
print(df.head(20))


# In[47]:


print(desc1)
print(desc2)


# In[32]:


# Long form --> Wide form : p pd.pivot_table
df_org = pd.pivot_table(df, index = ['id'], columns=['year'], values = ['A', 'B'], aggfunc='sum')
print(df_org)


# In[33]:


df_org.columns = [''.join(map(str, col)) for col in df_org.columns.values]
df_org = df_org.reset_index()
print(df_org)


# In[6]:


# 행렬 y, X 만들기
import wooldridge as woo
import numpy as np
import pandas as pd
import patsy as pt

gpa1 = woo.dataWoo('gpa1')
display(gpa1)

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


# In[9]:


# 데이터 세트를 훈련데이터(training)와 테스트 (test)데이터 세트로 나누기(무작위 20%내외)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 시계열 자료일 때는 뒷부분의 일부가 테스트 데이터세트로 사용됨 
# test = data.loc['2020-03-31':]
# train = data.loc[:'2019-12-31']

# test = data.iloc[:-12]
# train = data.iloc[-12:]


# ## 7.2 변수의 변환(Variables Transformation)

# ### 17.2.1 크기(scale) 조정 

# In[32]:


# 라이브러리 sklearn 불러오기

from sklearn import preprocessing 

# 데이터 불러오기
data = pd.read_csv('Data/insurance.csv')
display(data.head())

# 최대 최소값을 이용한 데이터 크기 조정
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))  

data['MM_age'] = minmax_scale.fit_transform(data[['age']])  
data['MM_bmi'] = minmax_scale.fit_transform(data[['bmi']])  
data['MM_children'] = minmax_scale.fit_transform(data[['children']])  
data['MM_charges'] = minmax_scale.fit_transform(data[['charges']])  

# 정규분포를 이용한 변수변환
stand_scale = preprocessing.StandardScaler()  

data['SS_age'] = stand_scale.fit_transform(data[['age']])  
data['SS_bmi'] = stand_scale.fit_transform(data[['bmi']])  
data['SS_children'] = stand_scale.fit_transform(data[['children']])  
data['SS_charges'] = stand_scale.fit_transform(data[['charges']])  

# Robust 표준화
robust_scale = preprocessing.RobustScaler()  

data['RS_age'] = robust_scale.fit_transform(data[['age']])  
data['RS_bmi'] = robust_scale.fit_transform(data[['bmi']])  
data['RS_children'] = robust_scale.fit_transform(data[['children']])  
data['RS_charges'] = robust_scale.fit_transform(data[['charges']])  

display(data.head())

display(data.describe().T)


# ### 17.2.4 다항변수 만들기

# In[4]:


# 데이터 불러오기
data = pd.read_csv('Data/insurance.csv')
display(data.head())

# 다항변수 만들기 
data['bmi^2'] = data['bmi']**2
data['bmi^3'] = data['bmi']**3
data


# ### 17.2.5 함수를 이용한 변수변환
# 

# In[89]:


# 데이터 불러오기
data = pd.read_csv('Data/insurance.csv')
display(data.head())

# 직접 변환
data['Log_charges'] = np.log(data['charges'])
data


# In[76]:


# 데이터 불러오기
data = pd.read_csv('Data/insurance.csv')
display(data.head())

# 함수를 이용한 변수변환 2
F = lambda x: np.log(x)
data['log_charges'] = data['charges'].apply(F)  

display(data.head()) 


# ### 17.2.6 이상치(outlier)의 탐지

# In[27]:


# 데이터 불러오기
data = pd.read_csv('Data/insurance.csv')

# 로그변환하여 아웃라이어를 축소
data['Log_charges'] = np.log(data['charges'])
display(data.head())

data.hist(bins=30, figsize=(10,6))
plt.tight_layout()
plt.show();


# In[28]:


data.boxplot(column=['age', 'bmi', 'children', 'charges'], figsize=(10,6))


# In[29]:


data.boxplot(column=['age', 'bmi', 'children', 'Log_charges'], figsize=(10,6))


# In[30]:


# interquartile range 1.5배 이상인 관측치는 이상치로 간주  
# 이상치 확인 함수  
def check_outliers(x):  
    q1, q3 = np.percentile(x, [25, 75])  
    iqr = q3-q1  
    lower_bound = q1 - (iqr * 1.5)  
    upper_bound = q3 + (iqr * 1.5)  
    return np.where((x > upper_bound)|(x < lower_bound), -1, 1)  

# 함수실행 이상치 판단변수 생성(상하위범위 벗어나면 -1, 아니면 1의 값)
data['Q_outlier'] = check_outliers(data['charges'].values)  
data.head(50)


# In[31]:


print(np.sum(data['Q_outlier'] == 1))  


# In[32]:


# Elliptical envelope를 이용한 이상치   

from sklearn.covariance import EllipticEnvelope  
outlier_detector = EllipticEnvelope(contamination=.1)  

# Predict outliers  
outlier_detector.fit(data[['charges']]) 
data['E_outlier'] = outlier_detector.predict(data[['charges']])  
display(data.head(50))


# In[33]:


print(np.sum(data['E_outlier'] == 1))  


# ### 17.2.7 Dealing with Outliers 

# In[34]:


# 이상치 제거하기   
data_no = data[data['E_outlier']==1]
display(data_no.head(50)) 


# ### 17.2.8 Replacing Outliers 

# In[35]:


# 이상치(E_outlier=0)를 nan으로 변경  
data['E_outlier'].replace(-1, np.nan, inplace = True)  
display(data.head(50)) 


# In[36]:


# E_outlier=nan을 9999로 대체  
data['E_outlier'].fillna(9999, inplace = True)  
display(data.head(50)) 


# In[37]:


# E_outlier=9999을 nan으로 만들기
data['E_outlier'].replace(9999, 0, inplace = True)  
display(data.head(50)) 


# ### 17.2.9 이산화(binarizing) 및 다중 숫자화(Discretizing Variables 

# In[38]:


# Importing the necessary functions from scikit learn  
from sklearn.preprocessing import Binarizer 


# In[39]:


# 데이터 불러오기
data = pd.read_csv('Data/insurance.csv')
data.describe()


# In[40]:


# 이산화
binarizer = Binarizer(threshold=10000)  

data['BI_charges'] = binarizer.fit_transform(data[['charges']])  
display(data.head()) 


# In[41]:


# 다중숫자화 
data['DG_bmi'] = np.digitize(data['bmi'], bins=[24, 30, 36])  
display(data.head(50)) 


# ### 17.2.10 범주형 변수(Categorical Variables)

# In[165]:


# Importing the necessary functions from scikit learn  
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer 


# In[172]:


# 데이터 불러오기
data = pd.read_csv('Data/insurance.csv')

# 2개의 카테고리를 구분 
bin_encoder = LabelBinarizer()  

data['BI_sex'] = bin_encoder.fit_transform(data[['sex']])  
display(data.head()) 


# In[173]:


# LabelEncoder allows creating a value for each category  
lab_encoder = LabelEncoder()  

# Replacing the transformed data  
data['LI_smoker'] = lab_encoder.fit_transform(data['smoker'])  
data['LI_region'] = lab_encoder.fit_transform(data['region'])  
display(data.head()) 


# In[174]:


mapper = {"male":0, "female":1}  
data["sex"].replace(mapper, inplace = True)  
display(data) 


# In[132]:


# 범주변수(categories variables)를 더미(dummy variables)로 바꾸기 
# OneHotEncoder 함수 사용  
one_hot = OneHotEncoder()  
new_data = one_hot.fit_transform(data[['region']]).toarray()  
display(new_data) 


# In[133]:


# Adding the dummy variables to the original data  
new_labels = np.unique(data_2['region'])  
new_df = pd.DataFrame(new_data, columns = new_labels)  
data_2 = pd.concat([data_2, new_df.iloc[:,:-1]], axis=1)  
del data_2['region'], new_data, new_labels, new_df 
display(data_2.head()) 


# In[177]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer 

# 데이터 불러오기
data = pd.read_csv('Data/insurance.csv')
data


# In[183]:


Column_trans = ColumnTransformer(
  [("Scaling", StandardScaler(), ['age', 'bmi', 'charges']), 
   ("OneHot", OneHotEncoder(sparse=False), ['sex', 'smoker', 'region'])])

Column_trans.fit(data)
datatrans = Column_trans.transform(data)
df = pd.DataFrame(datatrans)
df


# In[184]:


# 데이터 불러와서 데이터세트 전체에 있는 카테고리 변수를 더미변수로 만들기
data = pd.read_csv('Data/insurance.csv')

data_dum = pd.get_dummies(data)
data_dum


# ## 17.3 시계열 자료관련  

# In[185]:


# 날짜변수 만들기(1) 
data = pd.read_csv('Data/Korea_GDP.csv')

new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index
display(data)


# In[186]:


# 성장률(증가율)(exponential growth rate): 전년동기 대비 증가율
data['Gr_gdp'] = 100 * (np.log(data['gdp']) - np.log(data['gdp'].shift(4)))
data['Gr_con'] = 100 * (np.log(data['con']) - np.log(data['con'].shift(4)))
data


# In[187]:


# 전년동기 퍼센트 변화(Percent change)
data['Pc_gdp'] = 100*data['gdp'].pct_change(periods=4)
data['Pc_con'] = 100*data['con'].pct_change(periods=4)
data


# In[188]:


# 차분
data['D1_gdp'] = data['gdp'].diff() 
data


# In[202]:


# 이동평균
data['SMA_gdp']    = data['gdp'].rolling(window=4).mean()

data[['gdp','SMA_gdp']].plot(figsize=(10,4))


# In[203]:


# 자료의 Frequency변경
#일별자료를 주별자료 변환하려면 resample 사용  
data_Yr = data.resample('Y').last()  
data_Yr

