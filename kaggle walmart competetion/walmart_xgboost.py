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


categories = trainData.groupby('DepartmentDescription').groups.keys()
C_range = range(1,len(categories))

dictionaries = dict(zip(categories,C_range))    

trainData['Department'] = trainData.DepartmentDescription.map(dictionaries)
del trainData['DepartmentDescription']

#testData['Department'] = testData.DepartmentDescription.map(dictionaries)
#del testData['DepartmentDescription']

#y = trainData['TripType']
#del trainData['TripType']
X = trainData
#colName = ['Weekday','Upc','ScanCount','DepartmentDescription']
#colName = ['DepartmentDescription']

#for col in colName:
#    X[col] = abs((X[col].apply(hash))%2**(16))
#%% different kinds of trips
d


#%%
#X.dropna(how='any')
print ("handle missing data")
X.fillna(X.mean(),inplace=True)
#%%
X_train = X.groupby("VisitNumber").agg({"FinelineNumber": pd.Series.nunique, "Upc": pd.Series.nunique,\
                                    'ScanCount':np.sum,\
                                    'Department':pd.Series.nunique})

#aa['Count'] = X.groupby('VisitNumber').count()['Weekday']

#ScanCount - the number of the given item 
#%%hat was purchased. A negative value indicates a product return.
X_train['Count']= X.groupby('VisitNumber').ScanCount.count()

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

#%% for the test data

X_test = testData
X_test['WeekdayNumber'] = X_test.Weekday.map(dictionary)
del X_test['Weekday']
testData['Department'] = testData.DepartmentDescription.map(dictionaries)
del testData['DepartmentDescription']
    
#X_test.dropna(how='any')
print ("handle missing data")
X_test.fillna(X.mean(),inplace=True)
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
yy = le.fit_transform(y)
#%%  10 folder to ten different folder





#%%
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
#gbm = xgb.train(params, dtrain, num_trees)



#%%
result = gbm.predict(xgb.DMatrix(test_X))
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/sample_submission.csv')
Id = submit['VisitNumber']
submit.iloc[:,1:] = result
submit.iloc[:,0] = Id
submit.to_csv('/Users/weizhi/Desktop/kaggle walmart competetion/12_10_xgb_features.csv',index = False)

#%% neural network
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder() 
yy = encoder.fit_transform(y.values).astype(np.int32)
#%% from sklearn.cross_validation import KFold

from sklearn.cross_validation import KFold

kf = KFold(X.shape[0], n_folds=10)
count = 0
for train, test in kf:
    
    print("%s %s"%(train,test))
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
           
           ('dropout3', DropoutLayer),

           ('dense3',DenseLayer), # 200
           
          ('dropout4', DropoutLayer),

            ('dense4',DenseLayer), # 100


           ('output', DenseLayer)]
           
net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, 18),
                 dense0_num_units =400,
                 
                 dropout_p=0.25,
                 
                 dense1_num_units=600,
                 dropout2_p=0.25,

                 dense2_num_units=400,
                 dropout3_p=0.25,

                 dense3_num_units=600,
                 dropout4_p=0.25,

                 dense4_num_units = 200, 
                 output_num_units=38,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.005,
                 update_momentum=0.9,
                 
                 eval_size=0.1,
                 verbose=1,
                 max_epochs=200) 

#del X_scaled['VisitNumberCount'] 
net0.fit(X.values,yy)




#%%
resultNet = net0.predict_proba(test_X)
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/sample_submission.csv')
Id = submit['VisitNumber']
submit.iloc[:,1:] = resultNet
submit.iloc[:,0] = Id
submit.to_csv('/Users/weizhi/Desktop/kaggle walmart competetion/deeplearning.csv',index = False)





