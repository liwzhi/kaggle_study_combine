# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:49:11 2015

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
weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Sunday', 'Saturday']

dictionary = dict(zip(weekday,date))    

trainData['WeekdayNumber'] = trainData.Weekday.map(dictionary)
del trainData['Weekday']


categories = trainData.groupby('DepartmentDescription').groups.keys()
C_range = range(1,len(categories))

dictionaries = dict(zip(categories,C_range))    

trainData['Department'] = trainData.DepartmentDescription.map(dictionaries)
del trainData['DepartmentDescription']

#testData['Department'] = testData.DepartmentDescription.map(dictionaries)
#del testData['DepartmentDescription']

#y = trainData['TripType']
#del trainData['TripType']
trainData.fillna(trainData.mean(),inplace=True)
#%% get the tf features 




#%% we have 39 documents
aa = trainData.groupby(['TripType']).groups



import pylab as plt

def tripDict(trainData,aa,index,col):
    tripTwo = trainData.iloc[aa[aa.keys()[index]]]
    dictDepartment2 = {}
    for item in tripTwo[col]:
        if item in dictDepartment2:
            dictDepartment2[item] +=1
        else:
            dictDepartment2[item] =1
    return dictDepartment2
    


def plotCurve(dictData):
    plt.plot(sorted(dictData.keys()),dictData.values())

#plt.figure()
#for i in range(4):
#    dictDepartment = tripDict(trainData,aa,i,'Department')    
#    plotCurve(dictDepartment)
#%%count the value of department 
def countItem(trainData,colName):
    Dict = {}   
    for index in range(trainData[colName].shape[0]):
        key = trainData[colName].iloc[index]
        if key in Dict:
            Dict[key] +=1
        else:
            Dict[key] = 1
    count = 0
    for key in Dict.keys():
        count +=Dict[key]
       
    
    trainData['Count' + colName] = trainData[colName]
    for key in Dict.keys():
        trainData['Count' + colName].replace(key,Dict[key]/float(count),inplace=True) 
    return trainData

countCol = [ 'Upc', 'ScanCount', 'FinelineNumber','WeekdayNumber', 'Department']

for item in countCol:
    print item

    trainData = countItem(trainData,item) # add the department count

#%% for the test data

#%% for the test data
testData = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/test.csv')

X_test = testData
X_test['WeekdayNumber'] = X_test.Weekday.map(dictionary)
del X_test['Weekday']
testData['Department'] = testData.DepartmentDescription.map(dictionaries)
del testData['DepartmentDescription']
    
#X_test.dropna(how='any')
print ("handle missing data")
X_test.fillna(X_test.mean(),inplace=True)    

for item in countCol:
    print item

    X_test = countItem(X_test,item) # add the department count    


#%%
bb = X_test.groupby(['VisitNumber']).groups


#%%
X = trainData
y = trainData['TripType']


#colName = ['Weekday','Upc','ScanCount','DepartmentDescription']
#colName = ['DepartmentDescription']

#for col in colName:
#    X[col] = abs((X[col].apply(hash))%2**(16))
del trainData['TripType']
del trainData['VisitNumber']

cc =trainData[countCol].T.to_dict().values()
test_cc = X_test[countCol].T.to_dict().values()
#%% dict to item
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()

dd = vec.fit_transform(cc).toarray()
test_dd = vec.transform(test_cc).toarray()

from scipy import stats
test_dd =  stats.zscore(test_dd)
dd =  stats.zscore(dd)


#%% combine the count
countCol = [ 'CountUpc', 'CountScanCount', 'CountFinelineNumber',\
    'CountWeekdayNumber', 'CountDepartment']

countFeature = trainData[countCol].values

train_dd = np.concatenate((dd,countFeature),axis=1)

countFeature_test = X_test[countCol].values
test_dd = np.concatenate((test_dd,countFeature_test),axis=1)

#%% get the 





#%%
#%%
import random
#del X['VisitNumber']
aa = X.groupby(['TripType']).groups
X_train = pd.DataFrame()

index = []

for key in aa.keys():
    if len(aa[key])>2000:
        index = index + random.sample(aa[key],2000)
    else:
        index = index + aa[key]
        
X_train = train_dd[index,:]
yy = y.iloc[index]

#%% training the classifcation 


#%% neural network
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder() 
yy = encoder.fit_transform(yy.values).astype(np.int32)

#%%
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
 
    
layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),#100
           
           ('dropout', DropoutLayer),

           ('dense1', DenseLayer), # 200
           
          ('dropout2', DropoutLayer),

           ('dense2',DenseLayer), # 400



           ('output', DenseLayer)]
           
net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, 10),
                 dense0_num_units =100,
                 
                 dropout_p=0.25,
                 
                 dense1_num_units=300,
                 dropout2_p=0.25,

                 dense2_num_units=100,
             #    dropout3_p=0.25,


                 output_num_units=38,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.005,
                 update_momentum=0.9,
                 
                 eval_size=0.1,
                 verbose=1,
                 max_epochs=1000) 
#%% z-score to the traindata                  
net0.fit(X_train,yy)

#%% for the test data 
bb = X_test.groupby(['VisitNumber']).groups




#%%
resultDict = {}

count = 0
result2 = np.zeros((len(bb),38))
for key in bb.keys():
    result = np.mean(net0.predict_proba(test_dd[bb[key],:]),axis=0)
    resultDict[key] = result
    result2[count,:] = result
    count+=1


submitResult = pd.DataFrame.from_dict(resultDict)
    
    



#%%
#resultNet = net0.predict_proba(test_X)
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/sample_submission.csv')
Id = submit['VisitNumber']
submit.iloc[:,1:] = result2
submit.iloc[:,0] = Id
submit.to_csv('/Users/weizhi/Desktop/kaggle walmart competetion/deeplearning2.csv',index = False)










#%% get the model

#%% put in the xgboost trees
from sklearn.cross_validation import train_test_split
import xgboost as xgb

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

yy = le.fit_transform(yy)
#%%  10 folder to ten different folder
#%%


from sklearn import metrics


def lossFunction(act,pred):
    return metrics.log_loss(act,pred)

#%%

    
    
    
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
          "subsample": 0.95,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=400
num_boost_round =1000
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

gbm = xgb.train(params, dtrain, num_trees)

gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=1500, \
  feval=logLoss)



    
    











#%%
resultDict = {}

#test_dd2 =  stats.zscore(test_dd)
count = 0
result2 = np.zeros((len(bb),38))
for key in bb.keys():
    result = np.mean(gbm.predict(xgb.DMatrix(test_dd[bb[key],:])),axis=0)
    resultDict[key] = result
    result2[count,:] = result
    count+=1


submitResult = pd.DataFrame.from_dict(resultDict)
    
    



#%%
#resultNet = net0.predict_proba(test_X)
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/sample_submission.csv')
Id = submit['VisitNumber']
submit.iloc[:,1:] = result2
submit.iloc[:,0] = Id
submit.to_csv('/Users/weizhi/Desktop/kaggle walmart competetion/xgboost_2.csv',index = False)



#%%
result = gbm.predict(xgb.DMatrix(dd))
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/sample_submission.csv')
Id = submit['VisitNumber']
submit.iloc[:,1:] = result
submit.iloc[:,0] = Id
submit.to_csv('/Users/weizhi/Desktop/kaggle walmart competetion/12_14_xgb_features.csv',index = False)





#%% different kinds of trips
import random
del X['VisitNumber']
aa = X.groupby(['TripType']).groups
X_train = pd.DataFrame()

index = []

for key in aa.keys():
    if len(aa[key])>3000:
        index = index + random.sample(aa[key],3000)
    else:
        index = index + aa[key]
        
X_train = X.iloc[index,:]



#%% rows to item
from sklearn.feature_extraction import DictVectorizer
#vec = DictVectorizer()
ff =X_test.T.to_dict().values()

X_test = vec.transform(ff)
#%%
bb = testData.groupby(['VisitNumber']).groups
#X_test = pd.DataFrame()

index = []

for key in bb.keys():
    if len(bb[key])>1:
        index = index + random.sample(bb[key],1)
    else:
        index = index + bb[key]
        
test = X_test[index,:]

#%%
del test['VisitNumber']

ID = X_test['VisitNumber']
del X_test['VisitNumber']

#%%
#X.dropna(how='any')
print ("handle missing data")
X.fillna(X.mean(),inplace=True)
#%%
X_train = X.groupby(["TripType"]).agg({"FinelineNumber": pd.Series.nunique, "Upc": pd.Series.nunique,\
                                    'ScanCount':np.sum,\
                                    'Department':pd.Series.nunique})

#aa['Count'] = X.groupby('VisitNumber').count()['Weekday']

#ScanCount - the number of the given item 
#%%hat was purchased. A negative value indicates a product return.
X_train['Count']= X.groupby(["TripType","VisitNumber"]).ScanCount.count()

#%% things sold
X_train['ScanCountSum'] = X.groupby('VisitNumber').ScanCount.apply(np.sum)
# day to buy things
#%% how many fine number count
X_train['FinelineNumberSum'] = X.groupby('VisitNumber').FinelineNumber.transform('count')
#%% count the UPC 
X_train['UpcSum'] = X.groupby('VisitNumber').Upc.transform('count')


#%% get the weekday number
X_train['WeekdayNumber'] = X.groupby('VisitNumber').WeekdayNumber.mean()
#a = X.groupby('TripType').WeekdayNumber.unique()



#sale_means = train.groupby('Store').mean().Sales
#sale_means.name = 'Sales_Means'
#
#df_train = df_train.join(sale_means,on='Store')
#df_test = df_test.join(sale_means,on='Store')


# get the visitnumber
X_train['VisitNumberCount']= X.groupby('VisitNumber').VisitNumber.mean()


#%% get the target
X_train['TripType'] = X.groupby('VisitNumber').TripType.mean()

y = X_train['TripType']
del X_train['TripType']


X_train.fillna(X.mean(),inplace=True)


#X_train1 = standard_scaler.fit_transform(aa)


#%% new test data

test = X_test.groupby("VisitNumber").agg({"FinelineNumber": pd.Series.nunique, "Upc": pd.Series.nunique,\
                                    'ScanCount':np.sum,\
                                    'Department':pd.Series.nunique})

#aa['Count'] = X.groupby('VisitNumber').count()['Weekday']

#ScanCount - the number of the given item 
#%%hat was purchased. A negative value indicates a product return.
test['Count']= X_test.groupby('VisitNumber').ScanCount.count()

#%% things sold
test['ScanCountSum'] = X_test.groupby('VisitNumber').ScanCount.apply(np.sum)
# day to buy things
#%% how many fine number count
test['FinelineNumberSum'] = X_test.groupby('VisitNumber').FinelineNumber.transform('count')
#%% count the UPC 
test['UpcSum'] = X_test.groupby('VisitNumber').Upc.transform('count')
test['WeekdayNumber'] = X_test.groupby('VisitNumber').WeekdayNumber.mean()

#%% get the visit number
test['VisitNumberCount']= X_test.groupby('VisitNumber').VisitNumber.mean()

#%% buying rate from different featurs
def featureGet(data):
    features = pd.DataFrame()
    features['DepartDiff'] = data['Department'] / (data['Count'])  # from different department
    features['ScanCountRate'] = data['ScanCount'] / (data['Count']) # scane from differentscan 
    features['FinelineNumberRate'] = data['FinelineNumber'] / (data['Count'])
    features['UpcNumberRate'] = data['Upc'] / (data['Count'])
    features['Fineline2NumberRate'] = data['FinelineNumberSum'] / (data['Count'])
    features['Upc2NumberRate'] = data['UpcSum'] / (data['Count'])
    features['UpcRatio'] = data['UpcSum'] / (data['Upc'])
    features['FinelineNumberRatio'] = data['FinelineNumber'] / (data['FinelineNumberSum'])
    features['kindDepart'] = data['Department'] / (data['FinelineNumber'])
    return features

train_feature = featureGet(X_train)

test_feature = featureGet(test)    

X = pd.concat([X_train, train_feature], axis=1)
#del X['VisitNumberCount']
test_X = pd.concat([test, test_feature], axis=1)


from sklearn import preprocessing
transform = preprocessing.MinMaxScaler()
X_scaled = transform.fit_transform(X)


#%% apply PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=10)

data = pca.fit_transform(X_scaled)






#%% data explore 
cc =trainData.groupby('VisitNumber').groups
bb =trainData.groupby('TripType').groups
dd=trainData.groupby('FinelineNumber').groups

trainData.iloc[cc[cc.keys()[290]],:].Department.nunique()

#%% put in the xgboost trees
from sklearn.cross_validation import train_test_split
import xgboost as xgb

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X = X_train
y = X.TripType
yy = le.fit_transform(yy)
#%%  10 folder to ten different folder
#%%
del X['TripType']
X_train, X_valid,y_train,y_valid = train_test_split(X, yy, test_size=0.12, random_state=10)


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
          "subsample": 0.95,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=400
num_boost_round =1000,
gbm = xgb.train(params, dtrain, num_trees)



#%%
result = gbm.predict(xgb.DMatrix(dd))
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/sample_submission.csv')
Id = submit['VisitNumber']
submit.iloc[:,1:] = result
submit.iloc[:,0] = Id
submit.to_csv('/Users/weizhi/Desktop/kaggle walmart competetion/12_14_xgb_features.csv',index = False)


#%% from sklearn.cross_validation import KFold

from sklearn.cross_validation import KFold

kf = KFold(X.shape[0], n_folds=10)
count = 0
for train, test in kf:
    
    print("%s %s"%(train,test))
#%%
