#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[3]:


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
from scipy.stats import uniform
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold

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
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings("ignore")

# Korean Fonts
mpl.rc('font', family='NanumGothic')
mpl.rc('axes', unicode_minus=False)


# In[3]:


filename = "Data/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
peek = data.head(20)
print(peek)

shape = data.shape
print(shape)

types = data.dtypes
print(types)

#set_option('display.width', 100)
pd.set_option('display.max_rows', 10)

#set_option('precision', 3)
pd.set_option('display.precision', 3)

description = data.describe()
print(description)

class_counts = data.groupby('class').size()
print(class_counts)

correlations = data.corr(method='pearson')
print(correlations)

skew = data.skew()
print(skew)


# In[27]:


data.hist()
pyplot.tight_layout()
pyplot.show()


# In[28]:


data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.tight_layout()
pyplot.show()


# In[29]:


data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.tight_layout()
pyplot.show()


# In[30]:


sns.boxplot(data=data)
pyplot.tight_layout()
pyplot.show()


# In[31]:


correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()


# In[32]:


scatter_matrix(data)


# In[33]:


pairgrid = sns.PairGrid(data=data)
pairgrid = pairgrid.map_offdiag(sns.scatterplot)
pairgrid = pairgrid.map_diag(pyplot.hist)


# In[34]:


pairgrid = sns.PairGrid(data=data)
pairgrid = pairgrid.map_upper(sns.scatterplot)
pairgrid = pairgrid.map_diag(pyplot.hist)
pairgrid = pairgrid.map_lower(sns.kdeplot)


# In[ ]:





# In[4]:


# 종속변수와 설명변수의 구분
df_array = data.values
y = df_array[:, 8]
X = df_array[:, 0:8]
print(y, X)


# In[7]:


#  최소-최대 스케일링(Min-Max Scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
minmax_X = scaler.fit_transform(X)
print(minmax_X)


# In[8]:


# 표준화(Standardization)
scaler = StandardScaler().fit(X)
stand_X = scaler.transform(X)
print(stand_X)


# In[9]:


# 강건 스케일링(Robust Scaling)
scaler = RobustScaler().fit(X)
Robust_X = scaler.transform(X)
print(Robust_X)


# In[10]:


# 이진화(Binarization)print(normal_X )
binarizer = Binarizer(threshold=10.0).fit(X)
binary_X = binarizer.transform(X)
print(binary_X )


# In[11]:


# 정규화(Normalization)
scaler = Normalizer().fit(X)
normal_X = scaler.transform(X)
print(normal_X )


# # 변수선택

# In[4]:


# 종속변수와 설명변수의 구분
df_array = data.values
y = df_array[:, 8]
X = df_array[:, 0:8]
print(y, X)


# In[43]:


# 변수(특성)선택(feature extraction)
# 단변량 변수 선택방법
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, y)
print(fit.scores_)
print(fit.get_support())

features = fit.transform(X)
print(features)


# In[40]:


columns = 'preg plas pres skin test mass pedi age'
feature_name = columns.split()
n_features = X.shape[1]
pyplot.barh(np.arange(n_features), fit.scores_, align='center')
pyplot.yticks(np.arange(n_features), feature_name)
pyplot.xlabel('특성 중요도')
pyplot.ylabel('특성')
pyplot.ylim(-1, n_features)
pyplot.show()


# In[47]:


# 분산 임계값을 이용하는 방법
selector = VarianceThreshold(threshold=0.2)
selector.fit(X)
print(selector.get_support())
X_new = selector.fit_transform(X)
print(X_new)


# In[50]:


# 축차적 변수선택(Recursive feature elimination)
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, n_features_to_select=4)
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


# In[51]:


model = LogisticRegression(solver='liblinear')
rfecv = RFECV(model, cv=5)
fit = rfecv.fit(X, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


# In[44]:


n_scores = len(rfecv.cv_results_["mean_test_score"])

pyplot.figure()
pyplot.xlabel("Number of features selected")
pyplot.ylabel("Mean test accuracy")
pyplot.errorbar(
    range(1, n_scores + 1),
    rfecv.cv_results_["mean_test_score"],
    yerr=rfecv.cv_results_["std_test_score"],
               )
pyplot.title("Recursive Feature Elimination \n with correlated features")
pyplot.show()


# In[33]:


# L1-based feature selection
lsvc = LinearSVC(C=0.1, penalty="l1", dual=False).fit(X, y)
print(lsvc.coef_)

model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new)
print(X_new.shape)


# In[54]:


model = Lasso(alpha=0.01)
model.fit(X, y)
print(model.coef_)

model = SelectFromModel(model, prefit=True)
X_new = model.transform(X)
print(X_new)
print(X_new.shape)


# In[5]:


# Tree-based feature selection
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
print(clf.feature_importances_)


# In[6]:


model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape 
print(X_new)


# In[7]:


model = RandomForestClassifier()
model.fit(X, y)
#feature_importances = model.feature_importances_model = RandomForestClassifier()
#model.fit(X, y)
feature_importances = model.feature_importances_
print(feature_importances)


# In[9]:


model = SelectFromModel(model, prefit=True)
X_new = model.transform(X)
X_new.shape 
print(X_new)


# In[58]:


# 주요인 분석
pca = PCA(n_components=3)
fit = pca.fit(X)
print(fit.explained_variance_ratio_)
print(fit.components_)


# In[ ]:





# In[10]:


test_size = 0.33
seed = 12345
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("Accuracy: %.3f%%" % (result*100.0))


# In[ ]:





# In[11]:


kfold = KFold(n_splits=10, random_state=12345, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[12]:


loocv = LeaveOneOut()
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[52]:


n_splits = 10
test_size = 0.33
seed = 12345
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# # 모형의 성능평가 지표

# In[5]:


kfold = KFold(n_splits=10, random_state=12345, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[6]:


results = cross_val_score(model, X, y, cv=kfold, scoring='neg_log_loss')
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))


# In[60]:


results = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=12345)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(y_test, predicted)
print(matrix)


# In[64]:


report = classification_report(y_test, predicted)
print(report)


# # Regression Metrics

# In[5]:


filename = 'Data/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
print(array)


# In[6]:


X = array[:,0:13]
y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()

results = cross_val_score(model, X, y, cv=kfold, scoring= 'neg_mean_absolute_error')
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))


# In[7]:


results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))


# In[8]:


results = cross_val_score(model, X, y, cv=kfold, scoring='r2')
print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))


# # ML Algorithm

# In[9]:


filename = "Data/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# 설명변수와 종속변수 정의
df_array = data.values
X = df_array[:,0:8]
y = df_array[:,8]


# In[10]:


model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[11]:


model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[12]:


model = KNeighborsClassifier()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[13]:


model = GaussianNB()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[14]:


model = DecisionTreeClassifier()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[15]:


model = SVC()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[ ]:





# # Regression Metrics

# In[16]:


filename = 'Data/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
print(array)


# In[17]:


X = array[:,0:13]
y = array[:,13]
kfold = KFold(n_splits=10, random_state=12345, shuffle=True)


# In[18]:


model = LinearRegression()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))


# In[19]:


model = Ridge()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[20]:


model = Lasso()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[21]:


model = ElasticNet()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[22]:


model = KNeighborsRegressor()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[23]:


model = DecisionTreeRegressor()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[24]:


model = SVR(gamma='auto')
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())


# In[ ]:





# In[3]:


filename = "./Data/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# separate array into input and output components
df_array = data.values
X = df_array[:,0:8]
y = df_array[:,8]


# In[34]:


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


# In[ ]:





# In[4]:


# 파이프파인 생성
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

# 모형의 평가
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[9]:


# 변수선정
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)

# 파이프파인 생성
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression(solver='liblinear')))
model = Pipeline(estimators)

# 모형의 평가
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# # 앙상블

# In[37]:


filename = "./Data/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# separate array into input and output components
df_array = data.values
X = df_array[:,0:8]
y = df_array[:,8]

kfold = KFold(n_splits=10, random_state=12345, shuffle=True)


# In[38]:


cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(estimator=cart, n_estimators=num_trees, random_state=12345)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[39]:


num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[41]:


num_trees = 100
max_features = 7
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[42]:


num_trees = 30
model = AdaBoostClassifier(n_estimators=num_trees, random_state=12345)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[43]:


num_trees = 100
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=12345)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# In[44]:


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


# # Tuning¶

# In[45]:


alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = RidgeClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(X, y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)


# In[46]:


param_grid = {'alpha': uniform()}
model = RidgeClassifier()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=3, random_state=7)
rsearch.fit(X, y)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)


# # Save

# In[4]:


filename = './Data/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

# Fit the model on 33%
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# save the model to disk
filename1 = './Model/finalized_model1.sav'
dump(model, open(filename1, 'wb'))


# In[48]:


# load the model from disk
loaded_model = load(open(filename1, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

