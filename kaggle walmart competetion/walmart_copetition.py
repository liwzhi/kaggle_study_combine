# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:55:28 2015
TripType - a categorical id representing the type of shopping trip the customer made. This is the ground truth that you are predicting. TripType_999 is an "other" category.
VisitNumber - an id corresponding to a single trip by a single customer
Weekday - the weekday of the trip
Upc - the UPC number of the product purchased
ScanCount - the number of the given item that was purchased. A negative value indicates a product return.
DepartmentDescription - a high-level description of the item's department
FinelineNumber - a more refined category for each of the products, created by Walmart
@author: weizhi
"""
import pandas as pd
import numpy as np
import cPickle

trainData = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/train.csv')
testData = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/test.csv')
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/sample_submission.csv')

#%% target
date = range(1,8)
weekday = ['Monday', 'Tuesday', 'Friday', 'Wednesday', 'Thursday', 'Sunday', 'Saturday']

dictionary = dict(zip(weekday,date))    

trainData['WeekdayNumber'] = trainData.Weekday.map(dictionary)
del trainData['Weekday']





#y = trainData['TripType']
#del trainData['TripType']
X = trainData
#colName = ['Weekday','Upc','ScanCount','DepartmentDescription']
colName = ['DepartmentDescription']

for col in colName:
    X[col] = abs((X[col].apply(hash))%2**(16))
#%%
X.dropna(how='any')
print ("handle missing data")
X.fillna(X.mean(),inplace=True)

#aa = X
#y = aa['TripType']
#
#del aa['TripType']


#%%
X_train = X.groupby('VisitNumber').mean()  
#aa['Count'] = X.groupby('VisitNumber').count()['Weekday']
X_train['Count']= X.groupby('VisitNumber').TripType.count()
y = X_train['TripType']
del X_train['TripType']


#%% 
#aa = X.groupby('VisitNumber').groups   
#X_new = pd.DataFrame(columns = X.keys())    
#for key in aa.keys():
#    X_new = X_new.append(X.iloc[aa[key],:].mean(),ignore_index=True)    
#%%    


#%%
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, RobustScaler


standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

X_train = robust_scaler.fit_transform(aa)
X_train1 = standard_scaler.fit_transform(aa)


#%% for the test data

X_test = testData
X_test['WeekdayNumber'] = X_test.Weekday.map(dictionary)
del X_test['Weekday']
for col in colName:
    X_test[col] = abs((X_test[col].apply(hash))%2**(16))
    
#%%    
print ("handle missing data")
X_test.fillna(X_test.mean(),inplace=True)


X_test = X_test.groupby('VisitNumber').mean()
X_test['Count'] = testData.groupby('VisitNumber').ScanCount.count()


#%%

standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

X_test = robust_scaler.fit_transform(bb)
X_test1 = standard_scaler.fit_transform(bb)

#%% get the output
groups = y.unique()
dictGroup = {}
minNumber = []
for i in range(len(y)):
    item = y.iloc[i]
    if item in dictGroup:
        dictGroup[item].append(i)
    else:
        dictGroup[item] = [i]

import numpy as np
y_train = []
index = []

for key in dictGroup:
    minNumber.append(len(dictGroup[key]))
    if len(dictGroup[key])>200:
        y_train.append(y.iloc[dictGroup[key][:200]].values)
        index.append(dictGroup[key][:100])
    else:
        y_train.append(y.iloc[dictGroup[key]].values)
        index.append(dictGroup[key])

        
flattened = [val for sublist in index for val in sublist]
   

X = X_train[flattened,:]
y_new = y.iloc[flattened]
#X_new1 = X_train[flattened,:]
#y_new = y.iloc[flattened]
#%%


def performence(clf,train,label,clfName):
    re = cross_validation.ShuffleSplit(train.shape[0],n_iter=10,test_size =0.25,random_state =43)
    
    aucList = []
    accuracyList = []
    for train_index, test_index in re:
        clf.fit(train.iloc[train_index,:],y.iloc[train_index])
        pre_y = clf.predict_proba(train.iloc[test_index,:])  # probablity to get the AUC
        aucList.append(roc_auc_score(y.iloc[test_index],pre_y[:,1]))
        y_pred = clf.predict(train.iloc[test_index,:]) # get the accuracy of model 
        accuracyList.append(accuracy_score(y.iloc[test_index],y_pred))  
    print 'the classifications is ' + clfName
    print ("The AUC score is %f"%(sum(aucList)/10.))
    print ("The model accuracy is %f"%(sum(accuracyList)/10.))
    print "confusion matrix"
    print (confusion_matrix(y.iloc[test_index],y_pred))

#%% training a model 
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import log_loss

from sklearn import grid_search
rf = ensemble.RandomForestClassifier(n_estimators = 400,random_state=43,class_weight="balanced",max_features=None,min_samples_leaf=10)

parameters = {'n_estimators':[200,400,600],'min_samples_leaf':[5,10],'max_features':['sqrt',None]}
#
#svr = svm.SVC()
clf1 = grid_search.GridSearchCV(rf,parameters)
#clf.fit(result,y)
clf1.fit(X,y_new)

rf = clf1.best_estimator_
cv = cross_validation.cross_val_score(rf,X_new,y_new,cv=10)
print ('random forest')
print (cv.mean())


# save the classifier
with open('/Users/weizhi/Desktop/kaggle walmart competetion//randomForest.pkl', 'wb') as fid:
    cPickle.dump(rf, fid)    

# load it again
#with open('my_dumped_classifier.pkl', 'rb') as fid:
 #   gnb_loaded = cPickle.load(fid)

#%%
#performence(rf,result,y,'gradient boosting')

performence(rf,pd.DataFrame(X_new),y_new,'random forest')


performence(rf,pd.DataFrame(X_new1),X_new,'random forest')
#%% linear regeresion

#%% classificatoin 
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search
clf = linear_model.LogisticRegression(solver='newton-cg',random_state=43)

parameters = {'penalty':['l2'],'C':[0.1,1]}

clf1 = grid_search.GridSearchCV(clf,parameters)

clf1.fit(X_new,y_new)
clf1.fit(X_train,y)
lb = clf1.best_estimator_

cv = cross_validation.cross_val_score(lb,X_new,y_new,cv=10)
print (cv.mean())

with open('/Users/weizhi/Desktop/kaggle walmart competetion//logisRegression.pkl', 'wb') as fid:
    cPickle.dump(lb, fid)    


#%%

performence(lb,pd.DataFrame(X_new),y_new,'logistic regression')


performence(lb,pd.DataFrame(X_new1),X_new,'logistic regression')


#%% ada


ada = ensemble.GradientBoostingClassifier(random_state=43)

parameters = {'n_estimators':[100,200,400,600],'min_samples_leaf':[2,5,10],'max_features':['sqrt',None],'subsample':[0.5,0.8,1]}
#
#svr = svm.SVC()
clf1 = grid_search.GridSearchCV(ada,parameters)
#clf.fit(result,y)
clf1.fit(X_new,y_new)

adarf = clf1.best_estimator_
cv = cross_validation.cross_val_score(ada,result,y,cv=10)
print ('ada')
print (cv.mean())




#%% performance 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn import grid_search

def performence(clf,train,label,clfName):
    re = cross_validation.ShuffleSplit(train.shape[0],n_iter=10,test_size =0.25,random_state =43)
    
    aucList = []
    accuracyList = []
    for train_index, test_index in re:
        clf.fit(train.iloc[train_index,:],y.iloc[train_index])
        pre_y = clf.predict_proba(train.iloc[test_index,:])  # probablity to get the AUC
        aucList.append(roc_auc_score(y.iloc[test_index],pre_y[:,1]))
        y_pred = clf.predict(train.iloc[test_index,:]) # get the accuracy of model 
        accuracyList.append(accuracy_score(y.iloc[test_index],y_pred))  
    print 'the classifications is ' + clfName
    print ("The AUC score is %f"%(sum(aucList)/10.))
    print ("The model accuracy is %f"%(sum(accuracyList)/10.))
    print "confusion matrix"
    print (confusion_matrix(y.iloc[test_index],y_pred))
#%% adaboost
    
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import log_loss

from sklearn import grid_search
rf = ensemble.RandomForestClassifier(n_estimators = 200,random_state=43,class_weight="balanced",max_features=None)

parameters = {'n_estimators':[100,200,400,600],'min_samples_leaf':[2,5,10],'max_features':['sqrt',None]}
#
#svr = svm.SVC()
clf1 = grid_search.GridSearchCV(rf,parameters)
#clf.fit(result,y)
clf1.fit(X_new,y_new)

rf = clf1.best_estimator_
cv = cross_validation.cross_val_score(rf,X_new,y_new,cv=10)
print ('random forest')
print (cv.mean())
#performence(rf,X_new,y,'gradient boosting')

performence(rf,pd.DataFrame(X_new),y,'randomr forest')


performence(rf,pd.DataFrame(X_new1),y,'randomr forest')

#%% ababoosting
#GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
#              max_depth=3, max_features='sqrt', max_leaf_nodes=None,
#              min_samples_leaf=10, min_samples_split=2,
#              min_weight_fraction_leaf=0.0, n_estimators=400,
#              presort='auto', random_state=43, subsample=1, verbose=0,
#              warm_start=False)
ada = ensemble.GradientBoostingClassifier(random_state=43)

parameters = {'n_estimators':[200,400],'min_samples_leaf':[10],'max_features':['sqrt',None],'subsample':[0.8,1]}
#
#svr = svm.SVC()
clf1 = grid_search.GridSearchCV(ada,parameters)
#clf.fit(result,y)
clf1.fit(X_new,y_new)

adarf = clf1.best_estimator_
cv = cross_validation.cross_val_score(ada,X_new,y_new,cv=10)
print ('ada')
print (cv.mean())

#performence(adarf,result,y,'gradient boosting')

#performence(adarf,pd.DataFrame(X_new),y_new,'gradient boosting')

#performence(adarf,pd.DataFrame(X_new1),y,'gradient boosting')

#%% SVM classication
from sklearn import svm, grid_search
#
parameters = {'kernel':('linear','rbf','poly'),'C':[1,10,100],'gamma':[0.0001,0.01,0.1]}
#
svr = svm.SVC(random_state=43,probability=True)
clf = grid_search.GridSearchCV(svr,parameters)
clf.fit(X_new,y_new)

svr = clf.best_estimator_



cv = cross_validation.cross_val_score(svr,X_new,y_new,cv=10)
print (cv.mean())
#
#
#cv = cross_validation.cross_val_score(svr,result,y,cv=10)
#print (cv.mean())
#
#
#cv = cross_validation.cross_val_score(svr,X_train1,y,cv=10)
#print (cv.mean())
#performence(svr,result,y,'SVM')
performence(svr,pd.DataFrame(X_new),y,'SVM')
performence(adarf,pd.DataFrame(X_new1),y,'SVM')




#%% linear model
clf = linear_model.LogisticRegression(solver='liblinear',random_state=43)

parameters = {'penalty':['l1','l2'],'C':[0.01,0.1,1,10]}

clf1 = grid_search.GridSearchCV(clf,parameters)

clf1.fit(X_new,y_new)
lb = clf1.best_estimator_

cv = cross_validation.cross_val_score(lb,X_new,y_new,cv=10)
print (cv.mean())
#%% xgboosting

import scipy as sp
def logloss(act, pred):
    """ Vectorised computation of logloss """
    
    #cap in official Kaggle implementation, 
    #per forums/t/1576/r-code-for-logloss
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    
    #compute logloss function (vectorised)
    ll = sum(   act*sp.log(pred) + 
                sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


from sklearn.cross_validation import train_test_split
import xgboost as xgb

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
yy = le.fit_transform(y)


X_train, X_valid,y_train,y_valid = train_test_split(X_train, yy, test_size=0.12, random_state=10)


dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_valid, y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# train a XGBoost tree
print("Train a XGBoost model")
params = {"objective": "multi:softprob",
          "num_class":38,
          "eta": 0.3,
          "max_depth": 10,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.8,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=1000
num_boost_round =1000,
gbm = xgb.train(params, dtrain, num_trees)

#%%
param = {'max_depth':2, 'eta':1, 'silent':1 }
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 2

# user define objective function, given prediction, return gradient and second order gradient
# this is loglikelihood loss


# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train



#%%

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)
    
    
print("Train xgboost model")

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.1,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.4,
          "min_child_weight": 6,
          "silent": 1,
          "thread": 1,
          "seed": 1301
          }
num_boost_round = 1500

print("Train a XGBoost model")
X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=1500, \
  feval=rmspe_xg)


print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)
# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
result.to_csv('/Users/weizhi/Desktop/kaggle_Rossmann/xgboost_Kaggle_1500_submission.csv', index=False)


clfList.append(gbm)













#%% validatoin 
from sklearn import cross_validation
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix

re = cross_validation.ShuffleSplit(X_new.shape[0],n_iter=10,test_size =0.25,random_state =43)

aucList = []
for train_index, test_index in re:
    gbm = xgb.train(params,xgb.DMatrix(X_new[train_index,:],yy[train_index]),num_trees)
    pre_y = gbm.predict(xgb.DMatrix(X_new[test_index]))
    aucList.append(accuracy_score(yy[test_index],pre_y))

print "xbgboost"
print sum(aucList)/10.


#%%

rf.fit(X_train,y)
result = clf.predict_proba(X_test)

result = rf.predict_proba(X_test)

#%% put in the deep learning

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder() 
yy = encoder.fit_transform(y).astype(np.int32)



#%% Deep Learning
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
 
    
layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
        #   ('dropout2', DropoutLayer),

           ('dense2',DenseLayer),
           ('dense3',DenseLayer),

           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, 6),
                 dense0_num_units=50,
                 
                 dropout_p=0.5,
                 
                 dense1_num_units=200,
             #    dropout_p1=0.3,

                 dense2_num_units=300,
                 dense3_num_units=100,

                 output_num_units=38,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.0005,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=1000)  
net0.fit(X_train,yy)

#%%
#result = gbm.predict_proba(xgb.DMatrix(X_test))
#%% ensemble all the models -- find the weights
#% log loss function
from sklearn.metrics import log_loss

from scipy.optimize import minimize
X_train = X_new
y_new = y
def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(y.iloc[test_index], final_prediction)
    
re = cross_validation.ShuffleSplit(X_new.shape[0],n_iter=10,test_size =0.25,random_state =43)

cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
bounds = [(0,1)]*3

weights = []
X_train = X_new
for train_index, test_index in re:
    aucList = []
    predictions = []
    gbm = xgb.train(params,xgb.DMatrix(X_new[train_index,:],yy[train_index]),num_trees)
    rf.fit(X_train[train_index,:],y_new.iloc[train_index])
    ada.fit(X_train[train_index,:],y_new.iloc[train_index])
   # svr.fit(X_train[train_index,:],y[train_index])
   # knn.fit(X_train[train_index,:],y[train_index])
    
  #  y_train = encoder.fit_transform(y[train_index]).astype(np.int32)

#    net0.fit(X_train[train_index,:],y_train)
 #   predictions.append(svr.predict_proba(X_train[test_index])[:,1])
    predictions.append(gbm.predict(xgb.DMatrix(X_new[test_index])))
    predictions.append(ada.predict_proba(X_new[test_index])[:,1])
    predictions.append(rf.predict_proba(X_new[test_index])[:,1])
  #  predictions.append(knn.predict_proba(X_train[test_index])[:,1])
 #   predictions.append(net0.predict_proba(X_train[test_index])[:,1])
 #   netAccuracy.append(roc_auc_score(y.iloc[test_index],net0.predict_proba(X_train[test_index])[:,1]))
    starting_values = [1./3]*len(predictions)

  #  train_eval_probs = 0.30*svr.predict_proba(X_train[test_index])[:,1] + 0.15*gbm.predict(xgb.DMatrix(result.iloc[test_index])) \
   #    + 0.15*ada.predict_proba(X_train[test_index])[:,1] + 0.35*rf.predict_proba(X_train[test_index])[:,1] \
    #   +  0.05*knn.predict_proba(X_train[test_index])[:,1]
    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))
    weights.append(res['x'])
  #  aucList.append(roc_auc_score(y.iloc[test_index],train_eval_probs))

print "Deep Learning - CNN"
print sum(netAccuracy)/10.











#%%
resultNet = net0.predict_proba(X_test)
result = gbm.predict(xgb.DMatrix(X_test))
result_lb = lb.predict_proba(X_test)
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/sample_submission.csv')
Id = submit['VisitNumber']
submit.iloc[:,1:] = result
submit.iloc[:,0] = Id
submit.to_csv('/Users/weizhi/Desktop/kaggle walmart competetion/gbm.csv',index = False)





