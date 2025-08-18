#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWDirectory\MyStock")
# 현재 작업공간(working directory)확인  
os.getcwd() 


# In[2]:


import pandas as pd
import numpy as np
import dowhy
import pydot
import pygraphviz
from dowhy import CausalModel

import warnings
warnings.filterwarnings("ignore")


# In[3]:


# 데이터 불러오기
df=pd.read_csv('Data/Salary_Data.csv')
df.info()

# 칼럼명의 공백을 _로 바꿈
df.rename(columns=lambda x:x.replace(' ', '_'),inplace=True)

# 숫자 변수의 누락치를 평균으로 채움
numeric = ['Age', 'Years_of_Experience', 'Salary']
for col in numeric:
    df[col].fillna(df[col].mean(), inplace=True)

# 카테고리 변수의 누락치를 최빈수로 채움 
categorical = ['Gender', 'Education_Level']
for col in categorical:
    mode_val = df[col].mode()[0] 
    df[col].fillna(mode_val, inplace=True)

# 교육수준의 PhD=1, 나머지는 0으로 바꿈
df['Education_Level'] = np.where(df['Education_Level'] == "PhD", 1, 0)

# 성별의 Male=1, Female=0로 바꿈
df['Gender'] = np.where(df['Gender'] == "Male", 1, 0)

# 모형에 포함시킬 변수만 선택
df=df[['Age', 'Gender', 'Education_Level', 'Years_of_Experience', 'Salary']]
print(df.describe().T)


# In[4]:


# 인과 그래프 
causal_graph = """
                digraph {Age; Gender; Education_Level; Years_of_Experience; Salary;
                Gender -> Education_Level -> Salary;
                Gender -> Salary;
                Age -> Years_of_Experience -> Salary;
                Age -> Education_Level;
                Age -> Salary;
                Gender->Salary;
                       }"""


# In[5]:


# 모형설정
model= CausalModel(data = df,
                   graph=causal_graph.replace("\n", " "),
                   treatment='Education_Level',
                   outcome='Salary')
model.view_model()


# In[6]:


# 식별
estimand = model.identify_effect()
print(estimand)


# In[7]:


# 추정
estimate= model.estimate_effect(
             identified_estimand=estimand,
             method_name='backdoor.linear_regression',
             confidence_intervals=True,
             test_significance=True
            )
print(f'Estimate of causal effect: {estimate}')


# In[8]:


# PCM 사용한 추정
estimate2= model.estimate_effect(
             identified_estimand=estimand,
             method_name='backdoor.propensity_score_matching',
            )
print(f'Estimate of causal effect: {estimate2}')


# In[9]:


# 반증(3가지 방법)
refutel_common_cause=model.refute_estimate(estimand, estimate,"random_common_cause")
print(refutel_common_cause)

refutel_common_cause=model.refute_estimate(estimand, estimate,"data_subset_refuter")
print(refutel_common_cause)

refutel_common_cause=model.refute_estimate(estimand, estimate,"placebo_treatment_refuter")
print(refutel_common_cause)


# In[10]:


# 반사실적 결과 계산
counterfactual = model.estimate_effect(
    identified_estimand=estimand,
    method_name="backdoor.linear_regression",
    target_units=lambda df: df[df['Education_Level'] == 0]
)
print("Counterfactual Outcome for untreated units: ", counterfactual.value)


# In[11]:


# 추정된 효과를 다양한 하위 그룹으로 나누어 분석
subgroups = df['Years_of_Experience'].quantile([0.20, 0.40, 0.6, 0.8]).values

for threshold in subgroups:
    subgroup_data = df[df['Years_of_Experience'] <= threshold]
    subgroup_model = CausalModel(
        data=subgroup_data,
        treatment='Education_Level',
        outcome='Salary',
        graph=causal_graph.replace("\n", " "),
    )
    subgroup_estimand = subgroup_model.identify_effect()
    subgroup_estimate = subgroup_model.estimate_effect(
        subgroup_estimand,
        method_name="backdoor.propensity_score_matching"
    )
    print(f"Causal Effect for confounder <= {threshold}: ", subgroup_estimate.value)


# In[12]:


# What-if 분석 수행: 특정 처리 여부 변경
new_df = df.copy()
new_df['Education_Level'] = 1 

# 인과 효과 추정 
# 모형설정
new_model = CausalModel(
    data=new_df,
    graph=causal_graph.replace("\n", " "),
    treatment='Education_Level',
    outcome='Salary',
                        )
# 모형식별
new_estimand = new_model.identify_effect()

# 모형추정
new_estimate = new_model.estimate_effect(new_estimand, method_name="backdoor.linear_regression")
print("Predicted Outcome under treatment for all: ", new_estimate.value)


# In[ ]:





# In[13]:


import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot


# In[14]:


import numpy as np
import pandas as pd
from lingam import DirectLiNGAM
from lingam.utils import make_dot

# DirectLiNGAM 모델 적합
model = DirectLiNGAM()
model.fit(df)

# 데이터프레임의 컬럼 이름을 labels로 사용
labels = df.columns.tolist() 

# 그래프 그리기   
make_dot(model.adjacency_matrix_, labels=labels)


# In[ ]:





# In[15]:


import numpy as np
import pandas as pd
from lingam import DirectLiNGAM
from lingam.utils import make_dot

# 데이터 불러오기
df=pd.read_csv('Data/korea_macro.csv')
df.info()


# In[16]:


df.drop(columns=['time'], inplace=True)
print(df)


# In[17]:


# DirectLiNGAM 모델 적합
model = DirectLiNGAM()
model.fit(df)

# 데이터프레임의 컬럼 이름을 labels로 사용
labels = df.columns.tolist() 

# 그래프 그리기   
make_dot(model.adjacency_matrix_, labels=labels)


# In[18]:


from causallearn.search.FCMBased import lingam
model = lingam.ICALiNGAM()
model.fit(df)

print(model.causal_order_)
print(model.adjacency_matrix_)

# 데이터프레임의 컬럼 이름을 labels로 사용
labels = df.columns.tolist() 

# 그래프 그리기   
make_dot(model.adjacency_matrix_, labels=labels)


# In[19]:


from causallearn.search.FCMBased import lingam
model = lingam.VARLiNGAM()
model.fit(df)

print(model.causal_order_)
print(model.adjacency_matrices_[0])
print(model.adjacency_matrices_[1])
print(model.residuals_)

# 데이터프레임의 컬럼 이름을 labels로 사용
labels = df.columns.tolist() 

# 그래프 그리기   
make_dot(model.adjacency_matrices_[1], labels=labels)


# In[20]:


from causallearn.search.FCMBased import lingam
model = lingam.RCD()
model.fit(df)

print(model.adjacency_matrix_)
print(model.ancestors_list_)

# 그래프 그리기   
make_dot(model.adjacency_matrix_, labels=labels)


# In[21]:


from causallearn.search.ConstraintBased.PC import pc
data = df.to_numpy()
# default parameters
cg = pc(data)

# or customized parameters
#cg = pc(data, alpha, indep_test, stable, uc_rule, uc_priority, mvpc, correction_name, background_knowledge, verbose, show_progress)

# visualization using pydot
cg.draw_pydot_graph()


# In[ ]:


# or save the graph
from causallearn.utils.GraphUtils import GraphUtils

pyd = GraphUtils.to_pydot(cg.G)
pyd.write_png('simple_test.png')


# In[ ]:





# # DAG를 이용한 인과추론

# In[1]:


# 예제 1: networkx를 이용한 DAG rmflrl
import networkx as nx
import matplotlib.pyplot as plt

# Create a DAG
dag = nx.DiGraph()

# Add nodes
dag.add_nodes_from(["Education", "Income", "Family Background"])

# Add edges (representing causal relationships)
dag.add_edges_from([
    ("Family Background", "Education"),
    ("Family Background", "Income"),
    ("Education", "Income")
                   ])

# Visualize the DAG
pos = nx.spring_layout(dag)
nx.draw(dag, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=15, font_weight='bold')
plt.title("Causal Relationship Visualization using DAG")
plt.show()


# In[3]:


# 예제 2: DoWhy를 이용한 인과추론
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

# 표본자료의 생성(sample data)
np.random.seed(42)
data = pd.DataFrame({
    'Family_Background': np.random.normal(0, 1, 1000),
    'Education': np.random.normal(0, 1, 1000),
    'Income': np.random.normal(0, 1, 1000)
                    })

# 변수들의 인과관계 정의(여기서 Family Background는 교란변수)
data['Education'] += 0.5 * data['Family_Background'] + np.random.normal(0, 0.1, 1000)
data['Income'] += 0.7 * data['Education'] + 0.3 * data['Family_Background'] + np.random.normal(0, 0.1, 1000)

# 인과모형 설정
model = CausalModel(
    data=data,
    treatment='Education',
    outcome='Income',
    common_causes=['Family_Background']
                    )


# In[6]:


# 인과효과의 식별
identified_model = model.identify_effect()
print(identified_model)


# In[7]:


# 인과효과 모형의 추정
estimate = model.estimate_effect(identified_model, method_name="backdoor.linear_regression")
print("Estimated causal effect of Education on Income:", estimate.value)


# In[8]:


# 추정결과의 반증(placebo_treatment_refuter 활용)
refutation = model.refute_estimate(
    identified_model, estimate, 
    "placebo_treatment_refuter"
                                  )
print(refutation)


# In[ ]:


refutation = model.refute_estimate(
    identified_model, estimate, 
    "random_common_cause"
                                   )
print(refutation)

refutation = model.refute_estimate(
    identified_model, estimate, 
    "data_subset_refuter"
                                  )
print(refutation)


# In[ ]:





# #  EconML사례분석

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
# 현재 작업공간(working directory)확인  
os.getcwd() 


# In[2]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from econml.dml import DML

from sklearn.linear_model import LinearRegression 

# 데이터세트 불러오기
data = pd.read_excel('Data/jtrain2.xlsx') 
X = data[['age', 'educ', 'black']] 
y = data['re78'] 
T = data['train'] 

# 단순기술통계량
data.describe()


# In[6]:


# 훈련 및 검증 데이터세트로 나누기 
X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(X, y, T, test_size=0.2, random_state=12345) 


# In[7]:


# DML 모형 초기화 
dml_model = DML(model_y=RandomForestRegressor(), 
    model_t=RandomForestRegressor(), 
    model_final=LinearRegression(), 
    random_state=12345) 

# 모형추정
dml_model.fit(Y=y_train, T=T_train, X=X_train) 

# 처치효과(treatment effect) 추정
treatment_effects = dml_model.effect(X_test) 

print("Estimated treatment effects:", treatment_effects)
print("Estimated treatment effects(mean):", treatment_effects.mean())


# In[8]:


import matplotlib.pyplot as plt 
# 추정된 처치효과의 히스토그램 
plt.figure(figsize=(10, 6)) 
plt.hist(treatment_effects, bins=30, edgecolor='k', alpha=0.7) 
plt.title('Distribution of Estimated Treatment Effects')
plt.xlabel('Estimated Treatment Effect') 
plt.ylabel('Frequency') 
plt.show() 


# In[ ]:





# In[12]:


# 이중 견고 학습(Doubly Robust Learning)
from econml.dr import LinearDRLearner
# DRL 모형 초기화 
drl_model = LinearDRLearner()

# 모형추정
drl_model.fit(Y=y_train, T=T_train, X=X_train) 

# 처치효과(treatment effect) 추정
treatment_effects = drl_model.effect(X_test) 
print("Estimated treatment effects:", treatment_effects)
print("Estimated treatment effects(mean):", treatment_effects.mean())


# In[11]:


# 추정된 처치효과의 히스토그램 
plt.figure(figsize=(10, 6)) 
plt.hist(treatment_effects, bins=30, edgecolor='k', alpha=0.7) 
plt.title('Distribution of Estimated Treatment Effects')
plt.xlabel('Estimated Treatment Effect') 
plt.ylabel('Frequency') 
plt.show() 


# In[17]:


# Forest based 모형 
import sklearn
from econml.orf import DMLOrthoForest, DROrthoForest

# Forest based 모형 초기화 
causal_forest = DMLOrthoForest(n_trees=1, max_depth=1, subsample_ratio=1,
                     model_T=sklearn.linear_model.LinearRegression(),
                     model_Y=sklearn.linear_model.LinearRegression())

# 모형추정
causal_forest.fit(Y=y_train, T=T_train, X=X_train, W=X_train) 

# 처치효과(treatment effect) 추정
treatment_effects = causal_forest.effect(X_test) 
print("Estimated treatment effects:", treatment_effects)
print("Estimated treatment effects(mean):", treatment_effects.mean())


# In[18]:


# 추정된 처치효과의 히스토그램 
plt.figure(figsize=(10, 6)) 
plt.hist(treatment_effects, bins=30, edgecolor='k', alpha=0.7) 
plt.title('Distribution of Estimated Treatment Effects')
plt.xlabel('Estimated Treatment Effect') 
plt.ylabel('Frequency') 
plt.show() 


# # Meta-learner

# In[27]:


from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

# S learner 초기화
overall_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(445/100))
S_learner = SLearner(overall_model=overall_model)

# 모형의 추정
S_learner.fit(Y=y_train, T=T_train, X=X_train) 

# 처치효과(treatment effects) 추정
treatment_effects_S = S_learner.effect(X_test)

print("Estimated treatment effects_S:", treatment_effects_T)
print("Estimated treatment effects_S(mean):", treatment_effects_T.mean())

# 추정된 처치효과의 히스토그램 
plt.figure(figsize=(10, 6)) 
plt.hist(treatment_effects_S, bins=30, edgecolor='k', alpha=0.7) 
plt.title('Distribution of Estimated Treatment Effects')
plt.xlabel('Estimated Treatment Effect') 
plt.ylabel('Frequency') 
plt.show() 


# In[26]:


# T learner 초기화
models = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(445/100))
T_learner = TLearner(models=models)

# 모형의 추정
T_learner.fit(Y=y_train, T=T_train, X=X_train) 

# 처치효과(treatment effects) 추정
treatment_effects_T = T_learner.effect(X_test)

print("Estimated treatment effects_T:", treatment_effects_T)
print("Estimated treatment effects_T(mean):", treatment_effects_T.mean())

# 추정된 처치효과의 히스토그램 
plt.figure(figsize=(10, 6)) 
plt.hist(treatment_effects_T, bins=30, edgecolor='k', alpha=0.7) 
plt.title('Distribution of Estimated Treatment Effects')
plt.xlabel('Estimated Treatment Effect') 
plt.ylabel('Frequency') 
plt.show() 


# In[29]:


# X learner 초기화
model = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(445/100))
propensity_model = RandomForestClassifier(n_estimators=100, max_depth=6,
                                          min_samples_leaf=int(445/100))
X_learner = XLearner(models=model, propensity_model=propensity_model)

# 모형의 추정
X_learner.fit(Y=y_train, T=T_train, X=X_train) 

# 처치효과(treatment effects) 추정
treatment_effects_X = X_learner.effect(X_test)

print("Estimated treatment effects_X:", treatment_effects_T)
print("Estimated treatment effects_X(mean):", treatment_effects_T.mean())

# 추정된 처치효과의 히스토그램 
plt.figure(figsize=(10, 6)) 
plt.hist(treatment_effects_X, bins=30, edgecolor='k', alpha=0.7) 
plt.title('Distribution of Estimated Treatment Effects')
plt.xlabel('Estimated Treatment Effect') 
plt.ylabel('Frequency') 
plt.show() 


# # DoWhy와 EconML의 결합

# In[ ]:





# In[30]:


from copy import deepcopy
import json
import time

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import mean_absolute_percentage_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import dowhy
from dowhy import CausalModel

from econml.metalearners import SLearner, XLearner, TLearner
from econml.dml import LinearDML, CausalForestDML, DML
from econml.dr import DRLearner, SparseLinearDRLearner

from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from lightgbm import LGBMRegressor, LGBMClassifier

import networkx as nx

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import graphviz


# In[33]:


# 데이터세트 불러오기
data = pd.read_excel('Data/jtrain2.xlsx') 
df = data[['re78', 'train', 'age', 'educ', 'black']] 

# 단순기술통계량
df.describe()


# In[35]:


# 훈련 및 검증 데이터세트로 나누기 
df_train, df_test = train_test_split(df, test_size=0.2, random_state=12345) 
df_train


# In[38]:


# Construct the graph (the graph is constant for all iterations)
nodes = ['re78', 'train', 'age', 'educ', 'black']
edges = [
    ('train', 're78'),
    ('age', 're78'),
    ('age', 'train'),
    ('educ', 're78'),
    ('educ', 'train'),
    ('black', 're78'),
    ('black', 'train'),
        ]

# Graphic Modelling Language(GML)
gml_string = 'graph [directed 1\n'
for node in nodes:
    gml_string += f'\tnode [id "{node}" label "{node}"]\n'
for edge in edges:
    gml_string += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'
gml_string += ']'
gml_string


# In[41]:


# 인과모형(CausalModel) 정의
model = CausalModel(
    data=df_train,
    treatment='train',
    outcome='re78',
    effect_modifiers=['age', 'educ', 'black'],
    graph=gml_string
)
model.view_model()

# Get the estimand
estimand = model.identify_effect()
print(estimand)


# In[72]:


# EconML의 S-Learner 
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.SLearner',
    target_units='ate',
    method_params={
        'init_params': {'overall_model': LGBMRegressor(n_estimators=1000, max_depth=10)},
        'fit_params': {}
                   })
dir(estimate)
print(estimate.cate_estimates.mean())


# In[44]:


# EconML의 T-Learner
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.TLearner',
    target_units='ate',
    method_params={
        'init_params': {
        'models': [LGBMRegressor(n_estimators=1000, max_depth=10), LGBMRegressor(n_estimators=1000, max_depth=10)]
                       },
        'fit_params': {}
    })
print(estimate.cate_estimates.mean())


# In[45]:


# EconML에서 X-Learner 사용
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.XLearner',
    target_units='ate',
    method_params={
        'init_params': {
        'models': [LGBMRegressor(n_estimators=1000, max_depth=10), LGBMRegressor(n_estimators=1000, max_depth=10)]
        },
        'fit_params': {}
    })
print(estimate.cate_estimates.mean())


# In[57]:


# Get estimates using all learners
learners = [
    ('S-Learner', 'backdoor.econml.metalearners.SLearner', {'overall_model': LGBMRegressor(n_estimators=1000, max_depth=10)}),
    ('T-Learner', 'backdoor.econml.metalearners.TLearner', {'models': [LGBMRegressor(n_estimators=1000, max_depth=10), LGBMRegressor(n_estimators=1000, max_depth=10)]}),
    ('X-Learner', 'backdoor.econml.metalearners.XLearner', {'models': [LGBMRegressor(n_estimators=1000, max_depth=10), LGBMRegressor(n_estimators=1000, max_depth=10)]}),
    ('DR-Learner', 'backdoor.econml.dr.LinearDRLearner', {'model_propensity': LogisticRegression(), 'model_regression': LGBMRegressor(n_estimators=1000, max_depth=10)}),
    ('LinearDML', 'backdoor.econml.dml.LinearDML', {'model_y': LGBMRegressor(n_estimators=1000, max_depth=10), 'model_t': LogisticRegression()}),
    ('CausalForestDML', 'backdoor.econml.dml.CausalForestDML', {'n_estimators': 1000, 'max_depth': 10})
]
for learner_name, method_name, init_params in learners:
    estimate = model.estimate_effect(
        identified_estimand=estimand,
        method_name=method_name,
        target_units='ate',
        method_params={
            'init_params': init_params,
            'fit_params': {}
                       })
    
    print(f'{learner_name} estimated treatment effect: {estimate.cate_estimates.mean()}')

#Heterogeneous Treatment Effect and Meta Learnerse_estimates.mean()}')


# In[62]:


# Get estimates using all learners
learners = [
    ('S-Learner', 'backdoor.econml.metalearners.SLearner', {'overall_model': LGBMRegressor(n_estimators=1000, max_depth=10)}),
    ('T-Learner', 'backdoor.econml.metalearners.TLearner', {'models': [LGBMRegressor(n_estimators=1000, max_depth=10), LGBMRegressor(n_estimators=1000, max_depth=10)]}),
    ('X-Learner', 'backdoor.econml.metalearners.XLearner', {'models': [LGBMRegressor(n_estimators=1000, max_depth=10), LGBMRegressor(n_estimators=1000, max_depth=10)]}),
]
for learner_name, method_name, init_params in learners:
    estimate = model.estimate_effect(
        identified_estimand=estimand,
        method_name=method_name,
        target_units='ate',
        method_params={
            'init_params': init_params,
            'fit_params': {}
                       })
    
    print(f'{learner_name} estimated treatment effect: {estimate.cate_estimates.mean()}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
# 현재 작업공간(working directory)확인  
os.getcwd() 


# In[2]:


import pandas as pd
import numpy as np
import dowhy
import pydot
import pygraphviz
from sklearn.model_selection import train_test_split 
from dowhy import CausalModel
from econml.metalearners import SLearner, XLearner, TLearner
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore")


# In[3]:


# 데이터 읽어오기
data = pd.read_excel('Data/jtrain2.xlsx', usecols=['age', 'educ', 'black', 're78', 'train'])

# 단순기술통계량
data.describe()

# 훈련 및 검증 데이터세트로 나누기 
Train, Test = train_test_split(data, test_size=0.2, random_state=12345) 

display(Train)

display(Test)


# In[4]:


# 인과 그래프 
causal_graph = """
                digraph {age; educ; black; re78; train;
                train ->re78 ;
                age -> re78 ;
                age -> train;
                educ -> re78 ;
                educ -> train;
                black -> re78 ;
                black -> train;
                       }"""


# In[5]:


# Instantiate the CausalModel 
model = CausalModel(
    data=Train,
    treatment='train',
    outcome='re78',
    common_causes=['age', 'educ', 'black'],
    graph=causal_graph.replace("\n", " ")
                     )
model.view_model()


# In[6]:


# Get the estimand
estimand = model.identify_effect()
print(estimand)


# In[7]:


# EconML에서 LinearDML 사용
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV

estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.dml.LinearDML',
    target_units='ate',
    method_params={
        'init_params': {
            'model_y': RandomForestRegressor(),
            'model_t': RandomForestRegressor()
        },
        'fit_params': {}
    })
print(estimate.cate_estimates)


# In[8]:


# refutation tests을 통한 추정치의 강건성(Robustness) 확인 
res_random=model.refute_estimate(estimand, estimate, method_name="random_common_cause")
print(res_random)


# In[9]:


res_placebo=model.refute_estimate(estimand, estimate,
        method_name="placebo_treatment_refuter", placebo_type="permute",
        num_simulations=20)
print(res_placebo)


# In[ ]:


# Test 데이터을 이용한 처치효과 추정
cate_estimates_test = estimate.estimator.effect(df=Test)

print("Estimated treatment effects (Test):", cate_estimates_test)
print("Estimated treatment effects (Test mean):", cate_estimates_test.mean())


# In[18]:


# EconML에서 DR-Learner 사용
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.dr.LinearDRLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'model_propensity': LogisticRegression(),
            'model_regression': RandomForestRegressor()
        },
        'fit_params': {}
    })
print(estimate.cate_estimates)


# In[19]:


# refutation tests을 통한 추정치의 강건성(Robustness) 확인 
res_random=model.refute_estimate(estimand, estimate, method_name="random_common_cause")
print(res_random)


# In[17]:


res_placebo=model.refute_estimate(estimand, estimate,
        method_name="placebo_treatment_refuter", placebo_type="permute",
        num_simulations=20)
print(res_placebo)


# In[10]:


# Test 데이터을 이용한 처치효과 추정
cate_estimates_test = estimate.estimator.effect(df=Test)

print("Estimated treatment effects (Test):", cate_estimates_test)
print("Estimated treatment effects (Test mean):", cate_estimates_test.mean())


# In[55]:


estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.dml.CausalForestDML',
    target_units='ate',
    method_params={
        'init_params': {
        'n_estimators': 1000,
        'max_depth': 10
        },
        'fit_params': {}
    },
    X = Train[['age', 'educ', 'black']])
print(estimate.cate_estimates)


# In[11]:


# EconML의 S-Learner 
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.SLearner',
    target_units='ate',
    method_params={
        'init_params': {'overall_model': LGBMRegressor(n_estimators=1000, max_depth=10)},
        'fit_params': {}
                   })
dir(estimate)
print(estimate.cate_estimates.mean())


# In[12]:


# Test 데이터을 이용한 처치효과 추정
cate_estimates_test = estimate.estimator.effect(df=Test)

print("Estimated treatment effects (Test):", cate_estimates_test)
print("Estimated treatment effects (Test mean):", cate_estimates_test.mean())


# In[21]:


# refutation tests을 통한 추정치의 강건성(Robustness) 확인 
res_random=model.refute_estimate(estimand, estimate, method_name="random_common_cause")
print(res_random)


# In[39]:


# EconML의 T-Learner
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.TLearner',
    target_units='ate',
    method_params={
        'init_params': {
        'models': [LGBMRegressor(n_estimators=1000, max_depth=10), LGBMRegressor(n_estimators=1000, max_depth=10)]
                       },
        'fit_params': {}
    })
print(estimate.cate_estimates.mean())


# In[40]:


# EconML에서 X-Learner 사용
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.XLearner',
    target_units='ate',
    method_params={
        'init_params': {
        'models': [LGBMRegressor(n_estimators=1000, max_depth=10), LGBMRegressor(n_estimators=1000, max_depth=10)]
        },
        'fit_params': {}
    })
print(estimate.cate_estimates.mean())


# In[41]:


# Get estimates using all learners
learners = [
    ('S-Learner', 'backdoor.econml.metalearners.SLearner', {'overall_model': LGBMRegressor(n_estimators=1000, max_depth=10)}),
    ('T-Learner', 'backdoor.econml.metalearners.TLearner', {'models': [LGBMRegressor(n_estimators=1000, max_depth=10), LGBMRegressor(n_estimators=1000, max_depth=10)]}),
    ('X-Learner', 'backdoor.econml.metalearners.XLearner', {'models': [LGBMRegressor(n_estimators=1000, max_depth=10), LGBMRegressor(n_estimators=1000, max_depth=10)]}),
]
for learner_name, method_name, init_params in learners:
    estimate = model.estimate_effect(
        identified_estimand=estimand,
        method_name=method_name,
        target_units='ate',
        method_params={
            'init_params': init_params,
            'fit_params': {}
                       })
    
    print(f'{learner_name} estimated treatment effect: {estimate.cate_estimates.mean()}')


# In[ ]:





# In[ ]:




