#!/usr/bin/env python
# coding: utf-8

# In[29]:


import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from joblib import dump
from joblib import load
from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
from pickle import dump
from pickle import load
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings("ignore")

# Korean Fonts
mpl.rc('font', family='NanumGothic')
mpl.rc('axes', unicode_minus=False)


# In[51]:


df = read_csv('./Data/mroz.csv', usecols=['inlf', 'educ', 'exper', 'age', 'kidslt6', 'kidsge6'])
data = df.sample(frac=1.0, random_state=1234)
data['inlf'] = data['inlf'].astype('category')
display(data)


# In[52]:


shape = data.shape
print(shape)

types = data.dtypes
print(types)

#set_option('display.width', 100)
pd.set_option('display.max_columns', None)
#set_option('precision', 3)
pd.set_option('display.precision', 3)

description = data.describe()
print(description)

class_counts = data.groupby('kidslt6').size()
print(class_counts)

correlations = data.corr(method='pearson', numeric_only=True)
print(correlations)

skew = data.skew()
print(skew)


# In[13]:


data.hist()


# In[14]:


data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)


# In[15]:


data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)


# In[16]:


sns.boxplot(data=data)


# In[19]:


correlations = data.corr(numeric_only=True)
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)


# In[22]:


scatter_matrix(data)


# In[ ]:


pairgrid = sns.PairGrid(data=data)
pairgrid = pairgrid.map_offdiag(sns.scatterplot)
pairgrid = pairgrid.map_diag(pyplot.hist)


# In[ ]:


pairgrid = sns.PairGrid(data=data)
pairgrid = pairgrid.map_upper(sns.scatterplot)
pairgrid = pairgrid.map_diag(pyplot.hist)
pairgrid = pairgrid.map_lower(sns.kdeplot)


# In[53]:


# separate array into input and output components
df_array = data.values
X = df_array[:,1:6]
y = df_array[:,0]


# In[25]:


scaler = MinMaxScaler(feature_range=(0, 1))
minmax_X = scaler.fit_transform(X)
print(minmax_X)


# In[26]:


scaler = StandardScaler().fit(X)
stand_X = scaler.transform(X)
print(stand_X)


# In[27]:


scaler = Normalizer().fit(X)
normal_X = scaler.transform(X)
print(normal_X )


# In[28]:


binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)
print(binary_X )


# In[64]:


# feature extraction
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, y)
print(fit.scores_)

features = fit.transform(X)
print(features)


# In[65]:


feature_name = ['educ', 'exper', 'age', 'kidslt6', 'kidsge6']
n_features = X.shape[1]
pyplot.barh(np.arange(n_features), fit.scores_, align='center')
pyplot.yticks(np.arange(n_features), feature_name)
pyplot.xlabel('특성 중요도')
pyplot.ylabel('특성')
pyplot.ylim(-1, n_features)
pyplot.show()


# In[56]:


# feature extraction
model = LogisticRegression(solver='liblinear')

rfecv = RFECV(model, cv=5)
fit = rfecv.fit(X, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


# In[57]:


min_features_to_select = 1 
clf = LogisticRegression()
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
            )
rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")


# In[58]:


n_scores = len(rfecv.cv_results_["mean_test_score"])

pyplot.figure()
pyplot.xlabel("Number of features selected")
pyplot.ylabel("Mean test accuracy")
pyplot.errorbar(
    range(1, n_scores + 1),
    rfecv.cv_results_["mean_test_score"],
    yerr=rfecv.cv_results_["std_test_score"],
)
pyplot.title("Recursive Feature Elimination \nwith correlated features")
pyplot.show()


# In[59]:


# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)


# In[60]:


# feature extraction
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, y)
print(model.feature_importances_)


# In[62]:


feature_name = ['educ', 'exper', 'age', 'kidslt6', 'kidsge6']
n_features = X.shape[1]
pyplot.barh(np.arange(n_features), model.feature_importances_, align='center')
pyplot.yticks(np.arange(n_features), feature_name)
pyplot.xlabel('특성 중요도')
pyplot.ylabel('특성')
pyplot.ylim(-1, n_features)
pyplot.show()


# In[67]:


test_size = 0.33
seed = 12345
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("Accuracy: %.3f%%" % (result*100.0))


# In[68]:


kfold = KFold(n_splits=10, random_state=12345, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[69]:


loocv = LeaveOneOut()
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[70]:


n_splits = 10
test_size = 0.33
seed = 12345
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[ ]:





# In[71]:


kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'accuracy'
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[72]:


kfold = KFold(n_splits=10, random_state=12345, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'neg_log_loss'
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:





# # Regression 

# In[101]:


df = read_csv('./Data/mroz.csv', usecols=['inlf', 'lwage', 'educ', 'exper', 'city'])
data = df[(df['inlf']==1)]
display(data)
data.describe().T


# In[102]:


types = data.dtypes
print(types)


# In[103]:


array = data.values
X = array[:,1:4]
y = array[:,4]
kfold = KFold(n_splits=10, random_state=12345, shuffle=True)
model = LinearRegression()

results = cross_val_score(model, X, y, cv=kfold, scoring= 'neg_mean_absolute_error')
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))


# In[104]:


results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))


# In[105]:


results = cross_val_score(model, X, y, cv=kfold, scoring='r2')
print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))


# # ML Algorithm

# In[107]:


df = read_csv('./Data/mroz.csv', usecols=['inlf', 'educ', 'exper', 'age', 'kidslt6', 'kidsge6'])
data = df.sample(frac=1.0, random_state=1234)
data['inlf'] = data['inlf'].astype('category')
display(data)


# In[108]:


# separate array into input and output components
df_array = data.values
X = df_array[:,1:6]
y = df_array[:,0]


# In[109]:


model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[110]:


model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[111]:


model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[112]:


model = KNeighborsClassifier()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[113]:


model = GaussianNB()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[114]:


model = DecisionTreeClassifier()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[115]:


model = SVC()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# # Regression Metrics

# In[116]:


df = read_csv('./Data/mroz.csv', usecols=['inlf', 'lwage', 'educ', 'exper', 'city'])
data = df[(df['inlf']==1)]
display(data)
data.describe().T


# In[117]:


array = data.values
X = array[:,1:4]
y = array[:,4]
kfold = KFold(n_splits=10, random_state=12345, shuffle=True)


# In[118]:


model = LinearRegression()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))


# In[122]:


model = Ridge()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[123]:


model = Lasso()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[124]:


model = ElasticNet()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[125]:


model = KNeighborsRegressor()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[126]:


model = DecisionTreeRegressor()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[128]:


model = SVR(gamma='auto')
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[ ]:





# In[132]:


df = read_csv(./Data/mroz.csv', usecols=['inlf', 'educ', 'exper', 'age', 'kidslt6', 'kidsge6'])
data = df.sample(frac=1.0, random_state=1234)
data['inlf'] = data['inlf'].astype('category')
display(data)

# separate array into input and output components
df_array = data.values
X = df_array[:,1:6]
y = df_array[:,0]


# In[133]:


# prepare models
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

    # boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[136]:


# create pipeline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[ ]:





# # 앙상블

# In[9]:


df = read_csv('./Data/mroz.csv', usecols=['inlf', 'educ', 'exper', 'age', 'kidslt6', 'kidsge6'])
data = df.sample(frac=1.0, random_state=1234)
data['inlf'] = data['inlf'].astype('category')
display(data)


# In[11]:


# separate array into input and output components
df_array = data.values
X = df_array[:,1:6]
y = df_array[:,0]

kfold = KFold(n_splits=10, random_state=12345, shuffle=True)


# In[12]:


cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(estimator=cart, n_estimators=num_trees, random_state=12345)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[13]:


num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[14]:


num_trees = 100
max_features = 7
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[15]:


num_trees = 30
model = AdaBoostClassifier(n_estimators=num_trees, random_state=12345)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[16]:


num_trees = 100
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=12345)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[17]:


# create the sub models
estimators = []
model1 = LogisticRegression(solver='liblinear')
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC(gamma='auto')
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, y, cv=kfold)
print(results.mean())


# # Tuning

# In[24]:


alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = RidgeClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(X, y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)


# In[28]:


param_grid = {'alpha': uniform()}
model = RidgeClassifier()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=3, random_state=7)
rsearch.fit(X, y)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)


# # Save

# In[30]:


df = read_csv('./Data/mroz.csv', usecols=['inlf', 'educ', 'exper', 'age', 'kidslt6', 'kidsge6'])
data = df.sample(frac=1.0, random_state=1234)
data['inlf'] = data['inlf'].astype('category')
display(data)

# separate array into input and output components
df_array = data.values
X = df_array[:,1:6]
y = df_array[:,0]


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

# Fit the model on 33%
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# save the model to disk
filename2 = './Model/finalized_model2.sav'
dump(model, open(filename2, 'wb'))


# In[44]:


# load the model from disk
loaded_model = load(open(filename2, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# In[ ]:





# In[ ]:




