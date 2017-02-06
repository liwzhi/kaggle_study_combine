# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:00:47 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:46:15 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 01 09:20:42 2015

@author: Algorithm 001
"""

#!/usr/bin/python
from __future__ import print_function
'''
Public Score :  0.11727 (previous 0.11771)
Private Validation Score : [1199]	train-rmspe:0.104377	eval-rmspe:0.093786
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import pylab as plt
# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])

    # Label encode some features
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.apply(lambda x:x.year)

    data['Month'] = data.Date.apply(lambda x:x.month)
    data['Day'] = data.Date.apply(lambda x:x.day)

    data['DayOfWeek'] = data.Date.apply(lambda x:x.dayofweek)
    data['WeekOfYear'] = data.Date.apply(lambda x:x.weekofyear)

    # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    # Calculate time competition open time in months
    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)
    # Promo open time in months
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    return data

## Start of main script

print("Load the training, test and store data using pandas")
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(int),
         'PromoInterval': np.dtype(str)}

train = pd.read_csv('/Users/weizhi/Desktop/kaggle_Rossmann/train.csv')
test = pd.read_csv('/Users/weizhi/Desktop/kaggle_Rossmann/test.csv')
store = pd.read_csv('/Users/weizhi/Desktop/kaggle_Rossmann/store.csv')
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle_Rossmann/xgboost_Kaggle_1500_submission 3.csv')
train.Date = train.Date.astype(np.datetime64)
test.Date = test.Date.astype(np.datetime64)
#%% group the data

#store = pd.read_csv('C:/Users/Algorithm 001/Desktop/kaggle_sales data/store.csv')
#train = pd.read_csv('C:/Users/Algorithm 001/Desktop/kaggle_sales data/train.csv',low_memory=False)
#test = pd.read_csv('C:/Users/Algorithm 001/Desktop/kaggle_sales data/test.csv')


print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")



#train.fillna(1, inplace=True)

train = train[train["Open"] != 0]
#print("Use only Sales bigger then zero")
train = train[train["Sales"] >0]


train['Sales'] = train['Sales'].replace(0,1)
print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
train = build_features(features, train)
test = build_features([], test)
print(features)


#%%
features.append('Sales_mean')
sales_mean = train.groupby('Store').mean().Sales
sales_mean.name = 'Sales_mean'
train = train.join(sales_mean,on='Store')

test = test.join(sales_mean,on='Store')  
#%% add the each 
features.append('sales_median')

sales_median = train.groupby('Store').median().Sales

sales_median.name = 'sales_median'
train = train.join(sales_median,on='Store')

test = test.join(sales_median,on='Store') 

#%% adding promotion times
features.append('promotion_time')

promotion_time = train.groupby('Store').Promo.apply(np.sum)

promotion_time.name = 'promotion_time'
train = train.join(promotion_time,on='Store')

test = test.join(promotion_time,on='Store') 


#%%
#for storeID in sales_median.keys():
#    train[train['Store']==storeID].Sales.replace(0,sales_median[storeID])



#%% each customer
features.append('Customers_mean')

Customers_mean = train.groupby('Store').mean().Customers

Customers_mean.name = 'Customers_mean'
train = train.join(Customers_mean,on='Store')

test = test.join(Customers_mean,on='Store') 

#%% cacluate the max and min values 
features.append('sales_max')

sales_max = train.groupby('Store').max().Sales
sales_max.name = 'sales_max'
train = train.join(sales_max,on='Store')

test = test.join(sales_max,on='Store') 


#%% cacluate the max and min values 
features.append('sales_min')

sales_min = train.groupby('Store').min().Sales
sales_min.name = 'sales_min'
train = train.join(sales_min,on='Store')

test = test.join(sales_min,on='Store') 

#%%




#%% buy rate from each custmoer 

features.append('buying')

Customers_mean_buy = train.groupby('Store').mean().Sales/train.groupby('Store').mean().Customers

Customers_mean_buy.name = 'buying'
train = train.join(Customers_mean_buy,on='Store')

test = test.join(Customers_mean_buy,on='Store') 
#%% sampleing 
 
#%% calculate the T_SNE data set 

from sklearn.manifold import TSNE
model = TSNE(n_components=2,random_state=0,perplexity =500)


#%% mspe calculation 

print('training data processed')

def rmspe(y, yhat):
    return np.sqrt(np.mean(((y - yhat)/y) ** 2))

def rmspe_rf(yhat, y):
    y = np.expm1(y)
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)

print("Train xgboost model")
#%% combine the features from the hash coding
print("To new classifier")
X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
#%%  model valdiation

clfList = []
from sklearn.ensemble import ExtraTreesRegressor 

for i in range(1):
    n = (i+3)*100
    rf = ExtraTreesRegressor(n_estimators=n,random_state =3,n_jobs =-1)
    rf.fit(X_train[features],y_train)
    
    
    q = [i for i in zip(features,rf.feature_importances_)]
    q = pd.DataFrame(q,columns=['Feature_importance','Importance'],index=features)
    q.plot(kind='bar')
    
    y_hat = rf.predict(X_valid[features])
    
    rmspe_rf(y_hat,y_valid)
#%% 
    
#%% importance

q = [i for i in zip(X_train[features].keys(),rf.feature_importances_) ]

q = pd.DataFrame(q,columns = ['Feature_Names','Importance'],index=X_train[features].keys())

#############################################    
clfList.append(rf.fit(train[features],np.log1p(train.Sales)))
rf1 = pd.DataFrame({"Id":test['Id'],'Sales':np.expm1(rf.predict(test[features]))})
num_bins =100
plt.hist(rf_result['Sales'],num_bins)
rf_result.to_csv('/Users/weizhi/Desktop/kaggle_Rossmann/rf_Kaggle_1500_submission.csv', index=False)

#%% outputs

from sklearn.ensemble import RandomForestRegressor    

rf_foest = RandomForestRegressor(n_estimators=300,max_depth=10,random_state=43)
rf_foest.fit(X_train[features],y_train)
        
        
q = [i for i in zip(features,rf_foest.feature_importances_)]
q = pd.DataFrame(q,columns=['Feature_importance','Importance'],index=features)
q.plot(kind='bar')

y_hat = rf_foest.predict(X_valid[features])

rmspe_rf(y_hat,y_valid)


rf2 = pd.DataFrame({"Id":test['Id'],'Sales':np.expm1(rf_foest.predict(test[features]))})
num_bins =100
plt.hist(rf_result['Sales'],num_bins)
rf_result.to_csv('/Users/weizhi/Desktop/kaggle_Rossmann/rf_Kaggle_1500_submission.csv', index=False)

#%% ada boost tree
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(n_estimators=100,random_state=477)
ada.fit(X_train[features],y_train)

q = [i for i in zip(features,ada.feature_importances_)]
q = pd.DataFrame(q,columns=['Feature_importance','Importance'],index=features)
q.plot(kind='bar')

y_hat = ada.predict(X_valid[features])

rmspe_rf(y_hat,y_valid)
rf3 = pd.DataFrame({"Id":test['Id'],'Sales':np.expm1(ada.predict(test[features]))})
num_bins =100
plt.hist(rf_result['Sales'],num_bins)
rf_result.to_csv('/Users/weizhi/Desktop/kaggle_Rossmann/rf_Kaggle_1500_submission.csv', index=False)

#%% kernel ridge model

from sklearn.kernel_ridge import KernelRidge
kernel_ridge = KernelRidge(alpha=1)
#kernel_ridge.fit(X_train[features],y_train)
#%% neibors
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=9,weights = 'distance')
neigh.fit(X_train[features],y_train)

y_hat = neigh.predict(X_valid[features])

rmspe_rf(y_hat,y_valid)

#knn_result = neigh.predict(test[features])

knn_result = pd.DataFrame({"Id":test['Id'],'Sales':np.expm1(neigh.predict(test[features]))})

###########################
clfList.append(neigh.fit(train[features],np.log1p(train.Sales)))

#%% knn distance 

from sklearn.neighbors import KNeighborsRegressor
neigh_d = KNeighborsRegressor(n_neighbors=9,weights = 'distance')
neigh_d.fit(X_train[features],y_train)

y_hat = neigh_d.predict(X_valid[features])

rmspe_rf(y_hat,y_valid)

#knn_result = neigh.predict(test[features])

knn_d_result = pd.DataFrame({"Id":test['Id'],'Sales':np.expm1(neigh_d.predict(test[features]))})


clfList.append(neigh_d.fit(train[features],np.log1p(train.Sales)))

#%%
print('training data processed')

def rmspe(y, yhat):
    return np.sqrt(np.mean(((y - yhat)/y) ** 2))


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)
    
    
print("Train xgboost model")

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.02,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "min_child_weight": 6,
          "silent": 1,
          "thread": 1,
          "seed": 101
          }
num_boost_round = 3000

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
#%% nerual network


#%% for data visulization 
xgb_1000 = pd.read_csv("C:/Users/Algorithm 001/Desktop/kaggle_sales data//xgboost_Kaggle_1000_submission.csv")


plt.close('all')
plt.figure()
plt.hist(np.log1p(xgb_1000['Sales']),num_bins)
plt.title("Sales title from XGB")



plt.figure()
plt.hist(np.log1p(train['Sales']),num_bins)
plt.title("Sales title from training")

plt.figure()
plt.hist(np.log1p(rf_result['Sales']),num_bins)
plt.title("rf_result")


plt.figure()
plt.hist(np.log1p(knn_result['Sales']),num_bins)
plt.title("knn_result")

plt.figure()
plt.hist(np.log1p(knn_d_result['Sales']),num_bins)
plt.title("knn_d_result")


plt.figure()
plt.hist(np.log1p(resultFinal),num_bins)
plt.title("combine the regression outputs")

#############
np.log1p(rf_result['Sales']).describe()
np.log1p(train['Sales']).describe()
np.log1p(xgb_1000['Sales']).describe()
np.log1p(knn_result['Sales']).describe()

np.log1p(knn_d_result['Sales']).describe()


np.log1p(pd.DataFrame(resultFinal)).describe()

#%% ensemble learning for all results

clfList 
result = 0.15*clfList[-3].predict(test[features]) + 0.05*clfList[-2].predict(test[features]) \
            + 0.05*clfList[-1].predict(test[features]) + 0.75*clfList[5].predict(xgb.DMatrix(test[features]))



result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(result)})
result.to_csv("C:/Users/Algorithm 001/Desktop/kaggle_sales data//xgboost_Kaggle_1000_submission.csv", index=False)

#resultFinal = np.expm1(result)



result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(result)})
result.to_csv("C:/Users/Algorithm 001/Desktop/kaggle_sales data//regression_combine.csv", index=False)
score = 0.11771
#%% combine all features column 
featureNew = pd.DataFrame()
featureNew['Knn_d'] = clfList[-3].predict(train[features])

featureNew['Knn'] = clfList[-2].predict(train[features])


featureNew['rf'] = clfList[-1].predict(train[features])

featureNew['xgb'] = clfList[5].predict(xgb.DMatrix(train[features]))
featureNew['Sales'] = train.Sales

#%% feature predict 

featureTest = pd.DataFrame()
featureTest['Knn_d'] = clfList[-3].predict(test[features])

featureTest['Knn'] = clfList[-2].predict(test[features])


featureTest['rf'] = clfList[-1].predict(test[features])

c['xgb'] = clfList[5].predict(xgb.DMatrix(test[features]))

result = featureTest['xgb'].values*0.85 + featureTest['rf'].values*0.05 + featureTest['Knn'].values*0.05 +\
                + featureTest['Knn_d'].values*0.05

result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(result)})

result = 0.5*result['Sales'] + 0.5*rf_result['Sales']

result = pd.DataFrame({"Id": test["Id"], 'Sales': result['Sales']})

result.to_csv('/Users/weizhi/Desktop/kaggle_Rossmann/xgboost_Kaggle_1500_submission.csv', index=False)
score = 0.11771
#%%

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.1,
          "max_depth": 3,
          "subsample": 0.9,
          "colsample_bytree": 0.4,
          "min_child_weight": 6,
          "silent": 1,
          "thread": 1,
          "seed": 1301
          }
num_boost_round = 50




print("Train a XGBoost model")
X_train, X_valid = train_test_split(featureNew, test_size=0.12, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_valid, y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm_Ensemble = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=200, \
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

rf 
ada
rf_foest

gbm


result['Sales'] = np.expm1(gbm.predict(dtest)*0.85 + rf.predict(test[features])*0.05 + ada.predict(test[features])*0.05 \
                                    + rf_foest.predict(test[features])*0.05)


rf1['Sales']*0.05 + rf2['Sales']*0.05 + rf3['Sales']*0.05 + result['Sales']*0.85
result.to_csv('/Users/weizhi/Desktop/kaggle_Rossmann/xgboost_Kaggle_1500_submission.csv', index=False)
  
  
  
  
#%%
  
neigh = KNeighborsRegressor(n_neighbors=9,weights = 'distance')
neigh.fit(X_train, y_train)

y_hat = neigh.predict(X_valid)  
rmspe_rf(y_hat,y_valid)


#%%


result = gbm_Ensemble.predict(xgb.DMatrix(featureTest))


plt.figure()
plt.hist(result,100)
#resultFinal = np.expm1(result)



result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(result)})
submit.to_csv("/Users/weizhi/Desktop/kaggle_Rossmann//test.csv", index=False)




