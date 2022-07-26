#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import_data = pd.read_csv('weatherAUS.csv')


# In[ ]:


df = import_data
df.dtypes


# In[ ]:


df.RainTomorrow


# In[ ]:


print(len(df), '----> Number of Samples in the Dataset')
print(df['RainTomorrow'].isnull().sum(), '----> Number of Empty Target Values (RainTomorrow)')
print(df['RainToday'].isnull().sum(), '---> Number of Empty RainToday Values')


# In[ ]:


#Replacing with 0 and 1 for convenience
df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)


# In[ ]:


#Imputing Target Values
df['RainToday'] = df['RainToday'].fillna(df['RainToday'].mode()[0])
df['RainTomorrow'] = df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])


# In[ ]:


print(df['RainTomorrow'].isnull().sum())
print(df['RainToday'].isnull().sum())
print(len(df))


# In[ ]:


import matplotlib.pyplot as plt
df.RainTomorrow.value_counts(normalize = True).plot(kind='bar')
plt.title('Plot to view imbalance in this dataset')
plt.show()


# In[ ]:


df['Date'] = df['Date'].astype('category').cat.codes
df['Location'] = df['Location'].astype('category').cat.codes
df['WindGustDir'] = df['WindGustDir'].astype('category').cat.codes
df['WindDir9am'] = df['WindDir9am'].astype('category').cat.codes
df['WindDir3pm'] = df['WindDir3pm'].astype('category').cat.codes


# In[ ]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
ImputedDF = df.copy(deep=True) 
mice_imputer = IterativeImputer()
ImputedDF.iloc[:, :] = mice_imputer.fit_transform(df)


# In[ ]:


print(ImputedDF.isnull().sum())


# In[ ]:


#Removing Outliers
from scipy import stats
CorrDF = ImputedDF[(np.abs(stats.zscore(ImputedDF)) < 3).all(axis=1)]


# In[ ]:


#Calculating the correlation 
correlation = CorrDF.corr()

relmat = correlation.iloc[:,-1]


#relmat.columns = ["Features", "Correlation with RainTomorrow"]

#print(relmat["columns"])
#rel = np.asmatrix(relmat.to_numpy())
    
#print(np.sort(relmat))
    


# In[ ]:


#plt.figure(figsize=(16,5))
#plt.plot(relmat)
#plt.title("Spatio - Temporal Dependencies")
#ax.bar(relmat, color ='maroon')
#plt.xlabel("Features")
#plt.ylabel("Correlation")


import seaborn as sns

#Plotting the Correlation Heat map
plt.figure(figsize=(16, 10))
sns.heatmap(correlation, annot= True)
plt.show()


#Plotting the Correlation between RainTommorow and other features
abs(relmat).plot(kind='bar')
plt.title('Correlation')
plt.show()


# In[ ]:


#Normalization of data
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(CorrDF)
NormalizedDF = pd.DataFrame(scaler.transform(CorrDF), index = CorrDF.index, columns = CorrDF.columns)
NormalizedDF.columns.values


# In[ ]:


#SPLITTING TARGET from DATASET
target = NormalizedDF.iloc[:,-1]
data = NormalizedDF.iloc[:,:-1]


# In[ ]:


#OVERSAMPLING
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
data, target = oversample.fit_resample(data, target)


# In[ ]:


import matplotlib.pyplot as plt
target.value_counts(normalize = True).plot(kind='bar')
plt.title('Balanced Dataset After SMOTE')
plt.show()


# In[ ]:


# Split into test and train

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=12345)

#Calculation of Feature importance using XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
plt.barh(X_train.columns.values, xgb.feature_importances_)
plt.xlabel("XGBoost Feature Importance")


# In[ ]:


#Feature Selection Being Implemented
features = data[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']]


# In[ ]:


# Split into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=12345)


# In[ ]:


#XGBoost
from xgboost import XGBClassifier
params_xgb ={'n_estimators': 500,
            'max_depth': 16}
xgb = XGBClassifier(**params_xgb)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)



from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
F1Score = f1_score(y_test, y_pred)
AccuracyXGB = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
prec, recall, fscore, sup = precision_recall_fscore_support(y_test, y_pred, average='macro')
percent=100/y_pred.shape[0]
print("F1 Score: ",F1Score)
print("Accuracy: ",AccuracyXGB)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(xgb, X_test, y_test)

print("\nTrue Positive %: ", tp*percent, "%\nFalse Positive %: ", fp*percent,"%\nFalse Negative %: ",fn*percent, "%\nTrue Negative %: ", tn*percent,"%\n")
print("\nPrecision: ", prec, "\nRecall: ", recall,"\nF-Score: ",fscore, "\nSupport: ", sup)


# In[ ]:


#Parameter Tuning for Random Forest Classifier
from sklearn import ensemble
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
RForest = ensemble.RandomForestClassifier()

gs = GridSearchCV(RForest(), param_grid, verbose =1, cv = 2, n_jobs=-1)
gs_results_dt = gs.fit(X_train, y_train)
print(gs_results_dt.best_score_)
print(gs_results_dt.best_estimator_)


# In[ ]:


#Parameter Tuning for MLP Classifier
from sklearn.neural_network import MLPClassifier

parameters = {'solver': ['lbfgs'], 'max_iter': [500,800,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
gs = GridSearchCV(MLPClassifier(), parameters, verbose =1, n_jobs=-1)
gs_results_dt = gs.fit(X_train, y_train)
print(gs_results_dt.best_score_)
print(gs_results_dt.best_estimator_)


# In[ ]:


#Parameter Tuning for AdaBoost
from sklearn.ensemble import AdaBoostClassifier

param_grid = {'n_estimators' : [100,200]
             'learning_rate' : [0.001, 0.01, 0.1, 0.2, 0.5]
             }

gs = GridSearchCV(AdaBoostClassifier(), param_grid, verbose =1, n_jobs=-1)
gs_results_dt = gs.fit(X_train, y_train)
print(gs_results_dt.best_score_)
print(gs_results_dt.best_estimator_)


# In[ ]:


#Parameter tuning for GaussianNB
from sklearn.naive_bayes import GaussianNB

grid_params = {'var_smoothing':[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}

gs = GridSearchCV(GaussianNB(), grid_params, verbose = 1, n_jobs=-1)

gs_results_nb = gs.fit(X_train, y_train)

print(gs_results_nb.best_score_)
print(gs_results_nb.best_estimator_)


# In[ ]:


#Parameter Tuning for Logistic Regression
from sklearn.linear_model import LogisticRegression

grid_params = {'penalty':['l1', 'l2'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

gs = GridSearchCV(LogisticRegression(), grid_params, cv=5, n_jobs=-1)

gs_results_lr = gs.fit(X_train, y_train)

print(gs_results_lr.best_score_)
print(gs_results_lr.best_estimator_)
print(gs_results_lr.best_params_)


# In[ ]:


#Parameter Tuning for Decision Tree
grid_params = {'criterion': ['gini', 'entropy'],
'max_depth': [i for i in range(2,40,3)],
'min_samples_leaf': [i for i in range(2, 40, 3)]}

gs = GridSearchCV(DecisionTreeClassifier(), grid_params, verbose = 1, n_jobs=-1)

gs_results_dt = gs.fit(X_train, y_train)

print(gs_results_dt.best_score_)
print(gs_results_dt.best_estimator_)


# In[ ]:


#Parameter tuning for K Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

grid_params = {
    'n_neighbors': [i for i in range(10,50,2)],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}


gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=2)

gs_results = gs.fit(X_train, y_train)


# In[ ]:


#Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nbg = GaussianNB()
nbg.fit(X_train, y_train)
y_pred = nbg.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
F1Score = f1_score(y_test, y_pred)
AccuracyNB = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
prec, recall, fscore, sup = precision_recall_fscore_support(y_test, y_pred, average='macro')
percent=100/y_pred.shape[0]
print("F1 Score: ",F1Score)
print("Accuracy: ",AccuracyNB)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(nbg, X_test, y_test)

print("\nTrue Positive %: ", tp*percent, "%\nFalse Positive %: ", fp*percent,"%\nFalse Negative %: ",fn*percent, "%\nTrue Negative %: ", tn*percent,"%\n")
print("\nPrecision: ", prec, "\nRecall: ", recall,"\nF-Score: ",fscore, "\nSupport: ", sup)


# In[ ]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
F1Score = f1_score(y_test, y_pred)
AccuracyDT = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
prec, recall, fscore, sup = precision_recall_fscore_support(y_test, y_pred, average='macro')
percent=100/y_pred.shape[0]
print("F1 Score: ",F1Score)
print("Accuracy: ",AccuracyDT)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(dtc, X_test, y_test)

print("\nTrue Positive %: ", tp*percent, "%\nFalse Positive %: ", fp*percent,"%\nFalse Negative %: ",fn*percent, "%\nTrue Negative %: ", tn*percent,"%\n")
print("\nPrecision: ", prec, "\nRecall: ", recall,"\nF-Score: ",fscore, "\nSupport: ", sup)


# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression()
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
F1Score = f1_score(y_test, y_pred)
AccuracyLR = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
prec, recall, fscore, sup = precision_recall_fscore_support(y_test, y_pred, average='macro')
percent=100/y_pred.shape[0]
print("F1 Score: ",F1Score)
print("Accuracy: ",AccuracyLR)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(logr, X_test, y_test)

print("\nTrue Positive %: ", tp*percent, "%\nFalse Positive %: ", fp*percent,"%\nFalse Negative %: ",fn*percent, "%\nTrue Negative %: ", tn*percent,"%\n")
print("\nPrecision: ", prec, "\nRecall: ", recall,"\nF-Score: ",fscore, "\nSupport: ", sup)


# In[ ]:


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
F1Score = f1_score(y_test, y_pred)
AccuracyKNN = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
prec, recall, fscore, sup = precision_recall_fscore_support(y_test, y_pred, average='macro')
percent=100/y_pred.shape[0]
print("F1 Score: ",F1Score)
print("Accuracy: ",AccuracyKNN)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(knn, X_test, y_test)

print("\nTrue Positive %: ", tp*percent, "%\nFalse Positive %: ", fp*percent,"%\nFalse Negative %: ",fn*percent, "%\nTrue Negative %: ", tn*percent,"%\n")
print("\nPrecision: ", prec, "\nRecall: ", recall,"\nF-Score: ",fscore, "\nSupport: ", sup)


# In[ ]:


#MLP Classifier
from sklearn.neural_network import MLPClassifier
params_nn = {'hidden_layer_sizes': (30,30,30),
             'activation': 'logistic',
             'solver': 'lbfgs',
             'max_iter': 500}

multi_layer_perceptron = MLPClassifier()
multi_layer_perceptron.fit(X_train, y_train)
y_pred = multi_layer_perceptron.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
F1Score = f1_score(y_test, y_pred)
AccuracyMLP = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
prec, recall, fscore, sup = precision_recall_fscore_support(y_test, y_pred, average='macro')
percent=100/y_pred.shape[0]
print("F1 Score: ",F1Score)
print("Accuracy: ",AccuracyMLP)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(multi_layer_perceptron, X_test, y_test)

print("\nTrue Positive %: ", tp*percent, "%\nFalse Positive %: ", fp*percent,"%\nFalse Negative %: ",fn*percent, "%\nTrue Negative %: ", tn*percent,"%\n")
print("\nPrecision: ", prec, "\nRecall: ", recall,"\nF-Score: ",fscore, "\nSupport: ", sup)


# In[ ]:


#AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
F1Score = f1_score(y_test, y_pred)
AccuracyAB = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
prec, recall, fscore, sup = precision_recall_fscore_support(y_test, y_pred, average='macro')
percent=100/y_pred.shape[0]
print("F1 Score: ",F1Score)
print("Accuracy: ",AccuracyAB)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(ada, X_test, y_test)

print("\nTrue Positive %: ", tp*percent, "%\nFalse Positive %: ", fp*percent,"%\nFalse Negative %: ",fn*percent, "%\nTrue Negative %: ", tn*percent,"%\n")
print("\nPrecision: ", prec, "\nRecall: ", recall,"\nF-Score: ",fscore, "\nSupport: ", sup)


# In[ ]:


#Random Forest Classifier
from sklearn import ensemble

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

RForest = ensemble.RandomForestClassifier()

#Fitting the train dataset
train= RForest.fit(X_train, y_train)
         
#Predict the test dataset
y_pred = RForest.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
F1Score = f1_score(y_test, y_pred)
AccuracyRF = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
prec, recall, fscore, sup = precision_recall_fscore_support(y_test, y_pred, average='macro')
percent=100/y_pred.shape[0]
print("F1 Score: ",F1Score)
print("Accuracy: ",AccuracyRF)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(RForest, X_test, y_test)

print("\nTrue Positive %: ", tp*percent, "%\nFalse Positive %: ", fp*percent,"%\nFalse Negative %: ",fn*percent, "%\nTrue Negative %: ", tn*percent,"%\n")
print("\nPrecision: ", prec, "\nRecall: ", recall,"\nF-Score: ",fscore, "\nSupport: ", sup)


# In[ ]:


#Plotting Graph for model comparision
accuracy_scores = [AccuracyLR, AccuracyKNN, AccuracyNB, AccuracyDT, AccuracyRF, AccuracyMLP, AccuracyAB, AccuracyXGB]
model_name = ['Logistic Regression','KNN','Naive Bayes','Decision Tree','Random Forest','MLPerceptron','AdaBoost','XGBoost']

plt.subplots(figsize=(12,10))
sns.barplot(model_name, accuracy_scores, palette='Paired')
plt.title("Model Comparision",fontsize=13)
plt.xlabel("Model Name")
plt.ylabel("Accuracy")

