#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")

# 현재 작업공간(working directory)확인  
os.getcwd() 


# In[20]:


get_ipython().system('pip install causal-learn')


# In[2]:


import pandas as pd
import numpy as np
import dowhy
import pydot
import pygraphviz
import graphviz
import networkx as nx
from dowhy import CausalModel

import warnings
warnings.filterwarnings("ignore")


# In[3]:


np.set_printoptions(precision=3, suppress=True)
np.random.seed(12345)


# In[4]:


data_mpg = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original',
                   delim_whitespace=True, header=None,
                   names = ['mpg', 'cylinders', 'displacement',
                            'horsepower', 'weight', 'acceleration',
                            'model year', 'origin', 'car name'])
data_mpg.to_csv('Data/auto_mpg.csv')
data_mpg


# In[5]:


data_mpg.dropna(inplace=True)
data_mpg.drop(['model year', 'origin', 'car name'], axis=1, inplace=True)
print(data_mpg.shape)
data_mpg.head()
data = data_mpg.to_numpy()


# In[11]:


from causallearn.search.ConstraintBased.PC import pc

labels = [f'{col}' for i, col in enumerate(data_mpg.columns)]
data = data_mpg.to_numpy()

cg = pc(data)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(cg.G, labels=labels)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()


# In[12]:


from causallearn.search.ScoreBased.GES import ges

# default parameters
Record = ges(data)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(Record['G'], labels=labels)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()


# In[13]:


from causallearn.search.FCMBased import lingam
model = lingam.ICALiNGAM()
model.fit(data)

from causallearn.search.FCMBased.lingam.utils import make_dot
make_dot(model.adjacency_matrix_, labels=labels)


# In[15]:


def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d

def str_to_dot(string):
    '''
    Converts input string from graphviz library to valid DOT graph format.
    '''
    graph = string.strip().replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph


# In[16]:


# Obtain valid dot format
graph_dot = make_graph(model.adjacency_matrix_, labels=labels)

# Define Causal Model
model=CausalModel(
        data = data_mpg,
        treatment='mpg',
        outcome='weight',
        graph=str_to_dot(graph_dot.source))

# Identification
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# Estimation
estimate = model.estimate_effect(identified_estimand,
                                method_name="backdoor.linear_regression",
                                control_value=0,
                                treatment_value=1,
                                confidence_intervals=True,
                                test_significance=True)
print("Causal Estimate is " + str(estimate.value))


# In[17]:


from causallearn.utils.Dataset import load_dataset

data_sachs, labels = load_dataset("sachs")

print(data.shape)
print(labels)


# In[18]:


graphs = {}
graphs_nx = {}
labels = [f'{col}' for i, col in enumerate(labels)]
data = data_sachs

from causallearn.search.ConstraintBased.PC import pc

cg = pc(data)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(cg.G, labels=labels)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()


# In[19]:


from causallearn.search.ScoreBased.GES import ges

# default parameters
Record = ges(data)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(Record['G'], labels=labels)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()


# In[20]:


from causallearn.search.FCMBased import lingam
model = lingam.ICALiNGAM()
model.fit(data)

from causallearn.search.FCMBased.lingam.utils import make_dot
make_dot(model.adjacency_matrix_, labels=labels)


# In[21]:


# Obtain valid dot format
graph_dot = make_graph(model.adjacency_matrix_, labels=labels)

data_df = pd.DataFrame(data=data, columns=labels)

# Define Causal Model
model_est=CausalModel(
        data = data_df,
        treatment='pip2',
        outcome='pkc',
        graph=str_to_dot(graph_dot.source))

# Identification
identified_estimand = model_est.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand)

# Estimation
estimate = model_est.estimate_effect(identified_estimand,
                                method_name="backdoor.linear_regression",
                                control_value=0,
                                treatment_value=1,
                                confidence_intervals=True,
                                test_significance=True)
print("Causal Estimate is " + str(estimate.value))


# In[ ]:





# In[ ]:





# # LINGAM을 이용한 Causal Discovery

# In[12]:


get_ipython().system('pip install lingam')


# In[21]:


import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot

np.set_printoptions(precision=3, suppress=True)
np.random.seed(100)

X = pd.read_csv('http://www.causality.inf.ethz.ch/data/lucas0_train.csv')
X.head()


# In[24]:


X.describe().T


# In[14]:


model = lingam.DirectLiNGAM()
model.fit(X)


# In[20]:


make_dot(model.adjacency_matrix_, labels=list(X.columns))


# # causallearn의 다양한 시각화 기법 

# In[34]:


get_ipython().system('pip install causal-learn')


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import matplotlib.image as mpimg
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.Granger.Granger import Granger
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils

# 데이터 읽어오기
X = pd.read_csv('http://www.causality.inf.ethz.ch/data/lucas0_train.csv')
X.head()
data = X.to_numpy()
variable_names = list(X.columns)


# In[11]:


# 1. Constraint-based Causal Discovery Methods
## PC Algorithm Example
causal_graph_pc = pc(data)
causal_graph_pc.draw_pydot_graph(labels=variable_names) 

# Save the graph
pyd = GraphUtils.to_pydot(causal_graph_pc.G, labels=variable_names)
pyd.write_png('Figures/causal_graph_pc.png')


# In[12]:


## FCI Algorithm Example
causal_graph_fci, edges = fci(data)
pdy = GraphUtils.to_pydot(causal_graph_fci, labels=variable_names)
pdy.write_png('Figures/causal_graph_fci.png')


# In[13]:


# 2. Score-based Causal Discovery Methods
## GES Algorithm with BIC Score Example

causal_graph_ges = ges(data)

pyd = GraphUtils.to_pydot(causal_graph_ges['G'], labels=variable_names)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()

# Save the graph
pyd.write_png('Figures/causal_graph_ges.png')


# In[68]:


## Exact Search Example
from causallearn.search.ScoreBased.ExactSearch import exact_search
causal_graph_exact = exact_search(data)
causal_graph_exact.show()


# In[15]:


# 3. Causal Discovery Methods based on Constrained Functional Causal Models
## LiNGAM-based Methods Example
model = lingam.DirectLiNGAM(data)
model.fit(data)
print(model.adjacency_matrix_)

make_dot(model.adjacency_matrix_, labels=list(X.columns))


# In[18]:


# 4. Hidden Causal Representation Learning
## GIN Condition-based Method Example
G, K = GIN(data)

pyd = GraphUtils.to_pydot(G)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()


# In[17]:


# 5. Permutation-based Causal Discovery Methods
## GRaSP Algorithm Example
G = grasp(data)

pyd = GraphUtils.to_pydot(G, labels=variable_names)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()



# In[ ]:





# In[ ]:




