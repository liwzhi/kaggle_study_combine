

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


#train['Sales'] = train['Sales'].replace(0,train["Sales"].median())
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



#%% each customer
features.append('Customers_mean')

Customers_mean = train.groupby('Store').mean().Customers

Customers_mean.name = 'Customers_mean'
train = train.join(Customers_mean,on='Store')

test = test.join(Customers_mean,on='Store') 
#%%

features.append('Customers_median')

Customers_median = train.groupby('Store').median().Customers

Customers_median.name = 'Customers_median'
train = train.join(Customers_median,on='Store')

test = test.join(Customers_median,on='Store') 

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

#%% buy rate from each custmoer 

features.append('buying')

Customers_mean_buy = train.groupby('Store').mean().Sales/train.groupby('Store').mean().Customers

Customers_mean_buy.name = 'buying'
train = train.join(Customers_mean_buy,on='Store')

test = test.join(Customers_mean_buy,on='Store') 
#%% sampleing 
features.append('buying_rate')

Customers_median_buy = train.groupby('Store').median().Sales/train.groupby('Store').median().Customers

Customers_median_buy.name = 'buying_rate'
train = train.join(Customers_median_buy,on='Store')

test = test.join(Customers_median_buy,on='Store')  
#%% calculate the T_SNE data set 


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
          "max_depth": 8,
          "subsample": 0.85,
          "colsample_bytree": 0.7,
          "min_child_weight": 6,
          "silent": 1,
          "thread": 1,
          "seed": 101
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
test_probs = np.expm1(gbm.predict(dtest))



#%% meta bagging leanring
num_runs = 7

for jj in xrange(num_runs):
  print(jj) 
  X_train, X_valid = train_test_split(train, test_size=0.012, random_state=jj*3)

  num_boost_round = 700
  params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.2,
          "max_depth": 10,
          "subsample": 0.85 + jj*0.01,
          "colsample_bytree": 0.7,
          "min_child_weight": 6,
          "silent": 1,
          "thread": 1,
          "seed": jj*3,
          }
  gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=1500,feval=rmspe_xg)
  
  print("Validating")
  yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
  error = rmspe(X_valid.Sales.values, np.expm1(yhat))
  print('RMSPE: {:.6f}'.format(error))
  
  test_probs = test_probs + np.expm1(gbm.predict(dtest))




# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales': test_probs/(num_runs +1)})
result.to_csv('/Users/weizhi/Desktop/kaggle_Rossmann/new_xgboost_ensemble_7_submission.csv', index=False)


