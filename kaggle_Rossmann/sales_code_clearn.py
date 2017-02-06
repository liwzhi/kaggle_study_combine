# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:52:37 2015

@author: weizhi
"""

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
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle_Rossmann/sample_submission.csv')

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
train = train[train["Sales"] >=0]


train['Sales'] = train['Sales'].replace(0,1)
print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
train = build_features(features, train)
test = build_features([], test)
print(features)
#%% get the sales 


test['Sales'] = pd.read_csv('/Users/weizhi/Downloads/rf1 (1).csv')['Sales']


from sklearn import linear_model



# Create linear regression object
#regr = linear_model.LinearRegression()
regr = linear_model.Lasso(alpha=0.4)


regr.fit(train[features],train.Customers)

test['Customers'] = regr.predict(test[features])


features.append('Customers')

#%% buying rate
features.append('buyingRate')

test['buyingRate'] = test['Sales'] / test['Customers']
train['buyingRate'] = train['Sales'] / train['Customers']





#%%
features.append('Customers_mean')
Customers_mean = train.groupby('Store').mean().Customers
Customers_mean.name = 'Customers_mean'

train = train.join(Customers_mean,on='Store')


Customers_mean = test.groupby('Store').mean().Customers
Customers_mean.name = 'Customers_mean'
test = test.join(Customers_mean,on='Store')  




#%%
features.append('Sales_mean')

sales_mean = train.groupby('Store').mean().Sales
sales_mean.name = 'Sales_mean'
train = train.join(sales_mean,on='Store')


sales_mean = test.groupby('Store').mean().Sales
sales_mean.name = 'Sales_mean'
test = test.join(sales_mean,on='Store')


#%% add the each 
features.append('sales_median')

sales_median = train.groupby('Store').median().Sales

sales_median.name = 'sales_median'
train = train.join(sales_median,on='Store')

sales_median = test.groupby('Store').median().Sales

sales_median.name = 'sales_median'
test = test.join(sales_median,on='Store')

#%% adding promotion times
features.append('promotion_time')

promotion_time = train.groupby('Store').Promo.apply(np.sum)

promotion_time.name = 'promotion_time'
train = train.join(promotion_time,on='Store')

promotion_time = test.groupby('Store').Promo.apply(np.sum)

promotion_time.name = 'promotion_time'
test = test.join(promotion_time,on='Store')
#%% promotion time
features.append('promotion_rate')

train['promotion_rate'] = train['Sales'] / train['promotion_time']

test['promotion_rate'] = test['Sales'] / test['promotion_time']

#%% school holiday
features.append('holidaySum')

holidaySum = train.groupby('Store').SchoolHoliday.apply(np.sum)


holidaySum.name = 'holidaySum'
train = train.join(holidaySum,on='Store')

holidaySum = test.groupby('Store').SchoolHoliday.apply(np.sum)

holidaySum.name = 'holidaySum'
test = test.join(holidaySum,on='Store')
#%% buying rate from the holiday

features.append('holidayBuy')

train['holidayBuy'] = train['Sales'] / train['holidaySum']

test['holidayBuy'] = test['Sales'] / test['holidaySum']

#%% cacluate the max and min values 
features.append('sales_max')

sales_max = train.groupby('Store').max().Sales
sales_max.name = 'sales_max'
train = train.join(sales_max,on='Store')

sales_max = test.groupby('Store').max().Sales
sales_max.name = 'sales_max'
test = test.join(sales_max,on='Store')

#%% cacluate the max and min values 
features.append('sales_min')

sales_min = train.groupby('Store').min().Sales
sales_min.name = 'sales_min'
train = train.join(sales_min,on='Store')

sales_min = test.groupby('Store').min().Sales
sales_min.name = 'sales_min'
test = test.join(sales_min,on='Store')
#%%

#%% buy rate from each custmoer 

features.append('buying')

Customers_mean_buy = train.groupby('Store').mean().Sales/train.groupby('Store').mean().Customers

Customers_mean_buy.name = 'buying'
train = train.join(Customers_mean_buy,on='Store')

Customers_mean_buy = test.groupby('Store').mean().Sales/test.groupby('Store').mean().Customers

Customers_mean_buy.name = 'buying'
test = test.join(Customers_mean_buy,on='Store')#%% sampleing 
 
#%% calculate the T_SNE data set 

from sklearn.manifold import TSNE
model = TSNE(n_components=2,random_state=0,perplexity =500)


#%% do the 

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()

name = ['CompetitionDistance','Promo','Promo2','SchoolHoliday',
           'buying','promotion_time','promotion_time' ]

#train[features].fillna(train[features].mean,interface=True)
#X = poly.fit_transform(train[features])
#XTest = poly.transform(test[features])


#%%
X_train, X_valid = train_test_split(train[features], test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)

#%%
from sklearn.ensemble import RandomForestRegressor    

rf_foest = RandomForestRegressor(n_estimators=300,max_depth=10,random_state=43)
rf_foest.fit(X_train,y_train)
        
        
q = [i for i in zip(features,rf_foest.feature_importances_)]
q = pd.DataFrame(q,columns=['Feature_importance','Importance'],index=features)
q.plot(kind='bar')

y_hat = rf_foest.predict(X_valid)

#rmspe_rf(y_hat,y_valid)
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
num_boost_round = 200

print("Train a XGBoost model")
X_train, X_valid,y_train, y_valid = train_test_split(train[features], train.Sales,test_size=0.012, random_state=10)
y_train = np.log1p(y_train)
y_valid = np.log1p(y_valid)
dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_valid, y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=1500, \
feval=rmspe_xg)


print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)


num_runs = 20

for jj in xrange(num_runs):
  print(jj) 
  params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.02,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "min_child_weight": 6,
          "silent": 1,
          "thread": 1,
          "seed": num_runs*3,
          }
  gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=1500,feval=rmspe_xg)
  
  print("Validating")
  yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
  error = rmspe(X_valid.Sales.values, np.expm1(yhat))
  print('RMSPE: {:.6f}'.format(error))
  
  test_probs = test_probs + gbm.predict(dtest)




#%% get the outputs
print("Make predictions on the test set")

dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)
# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
result.to_csv('/Users/weizhi/Desktop/kaggle_Rossmann/xgboost_Kaggle_1500_submission.csv', index=False)




#%%








