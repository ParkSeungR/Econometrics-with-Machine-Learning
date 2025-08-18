#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")

# 현재 작업공간(working directory)확인  
os.getcwd() 


# In[3]:


import pandas as pd
import numpy as np
import dowhy
import pydot
import pygraphviz
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
import io
import matplotlib.image as mpimg
from dowhy import CausalModel
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.Granger.Granger import Granger
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils

import warnings
warnings.filterwarnings("ignore")

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
variable_names = list(data_mpg.columns)


# In[6]:


# 1. Constraint-based Causal Discovery Methods
## PC Algorithm Example
causal_graph_pc = pc(data)
causal_graph_pc.draw_pydot_graph(labels=variable_names) 

# Save the graph
pyd = GraphUtils.to_pydot(causal_graph_pc.G, labels=variable_names)
pyd.write_png('Figures/causal_graph_pc.png')


# In[7]:


## FCI Algorithm Example
causal_graph_fci, edges = fci(data)
pdy = GraphUtils.to_pydot(causal_graph_fci, labels=variable_names)
pdy.write_png('Figures/causal_graph_fci.png')


# In[8]:


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


# In[10]:


## Exact Search Example
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
causal_graph_exact = bic_exact_search(data)
causal_graph_exact.show()


# In[12]:


# 3. Causal Discovery Methods based on Constrained Functional Causal Models
## LiNGAM-based Methods Example
model = lingam.DirectLiNGAM(data)
model.fit(data)
print(model.adjacency_matrix_)

make_dot(model.adjacency_matrix_, labels=variable_names)


# In[13]:


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


# In[14]:


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




