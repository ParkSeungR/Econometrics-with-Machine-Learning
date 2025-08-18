#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")

# 현재 작업공간(working directory)확인  
os.getcwd() 


# # 0. Importing Pandas 

# In[ ]:


import pandas as pd
import numpy as np


# # 1. Loading Data 

# In[1]:


# 가상적 dataset
data = {
    'Date': pd.date_range(start='2023-01-01', end='2023-01-10'),
    'Product': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
    'Sales': [100, 150, 120, 80, 200, 110, 90, 130, 160, 75],
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'West', 'North', 'East', 'South']
       }
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)


# In[4]:


# CSV 파일 불러오기
df = pd.read_csv('Data/charity.csv')

# 처음 일부자료 출력
df.head()


# # 2. Exploring Data Basics

# In[7]:


# 데이터프레임의 기초정보
df.info()

# 단순기술통계
df.describe()


# # 3. Handling Missing Data

# In[8]:


# 결측치 제거
df.dropna()

# 결측치를 NA로 표기
df.fillna('NA')


# # 4. Selecting Columns

# In[10]:


# Select a single column
df['charity']

# Select multiple columns
df[['charity', 'income']]


# # 5. Filtering Data

# In[12]:


# Filter rows based on a condition
df[df['deps'] > 3]


# In[20]:


# Multiple conditions
df['high'] = (df['income'] > 11) & (df['deps'] > 3)
display(df)


# # 6. Sorting Data

# In[21]:


# Sort DataFrame by a column
df.sort_values(by='income', ascending=False)


# # 7. Grouping and Aggregating Data

# In[23]:


# Group data by a column and calculate mean
df.groupby('time')['charity'].sum()


# # 8. Applying Functions to Data

# In[ ]:


# Apply a function to each element in a column
df['Sales'].apply(lambda x: x * 2)


# # 9. Filtering Data

# In[ ]:





# # 11. Filtering Data

# In[ ]:





# # 12. Filtering Data

# In[ ]:





# # 13. Filtering Data

# In[ ]:





# # 14. Filtering Data

# In[ ]:





# # 15. Filtering Data

# In[ ]:





# # 16. Filtering Data

# In[ ]:





# # 17. Filtering Data

# In[ ]:





# # 18. Filtering Data

# In[ ]:





# # 19. Filtering Data

# In[ ]:





# # 20. Filtering Data

# In[ ]:





# # 21. Filtering Data

# In[ ]:





# # 22. Filtering Data

# In[ ]:





# # 23. Filtering Data

# In[ ]:





# # 24. Filtering Data

# In[ ]:





# # 25. Filtering Data

# In[ ]:





# # 26. Filtering Data

# In[ ]:





# # 27. Filtering Data

# In[ ]:





# # 28. Filtering Data

# In[ ]:





# # 29. Filtering Data

# In[ ]:





# # 30. Filtering Data

# In[ ]:





# In[ ]:




