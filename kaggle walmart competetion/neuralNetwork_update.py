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
#%%
X.dropna(how='any')
print ("handle missing data")
X.fillna(X.mean(),inplace=True)
#%%
X_train = X.groupby("VisitNumber").agg({"FinelineNumber": pd.Series.nunique, "Upc": pd.Series.nunique,\
                                    'ScanCount':np.sum,'WeekdayNumber':pd.Series.nunique,\
                                    'Department':pd.Series.nunique})

#aa['Count'] = X.groupby('VisitNumber').count()['Weekday']
X_train['Count']= X.groupby('VisitNumber').TripType.count()
X_train['TripType'] = X.groupby('VisitNumber').TripType.mean()

X_train['VisitNumberCount']= X.groupby('VisitNumber').VisitNumber.mean()

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
    
#%%    
print ("handle missing data")
X_test.fillna(X_test.mean(),inplace=True)

test = testData.groupby("VisitNumber").agg({"FinelineNumber": pd.Series.nunique, "Upc": pd.Series.nunique,\
                                    'ScanCount':np.sum,'WeekdayNumber':pd.Series.nunique,\
                                    'Department':pd.Series.nunique})

#aa['Count'] = X.groupby('VisitNumber').count()['Weekday']

test['Count'] = testData.groupby('VisitNumber').ScanCount.count()



test['VisitNumberCount']= testData.groupby('VisitNumber').VisitNumber.mean()



test.fillna(X.mean(),inplace=True)
#%%
del X_train['VisitNumberCount']
del test['VisitNumberCount']
#%% put in the deep learning

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder() 
yy = encoder.fit_transform(y.values).astype(np.int32)


#%% Deep Learning
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
 
    
layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),#100
           
     #      ('dropout', DropoutLayer),

           ('dense1', DenseLayer), # 200
           
      #    ('dropout2', DropoutLayer),

           ('dense2',DenseLayer), # 400
           
       #    ('dropout3', DropoutLayer),

           ('dense3',DenseLayer), # 200
           
        #   ('dropout4', DropoutLayer),

            ('dense4',DenseLayer), # 100


           ('output', DenseLayer)]
           
net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, 10),
                 dense0_num_units=100,
                 
            #     dropout1_p=0.25,
                 
                 dense1_num_units=200,
             #    dropout2_p=0.25,

                 dense2_num_units=400,
              #   dropout3_p=0.25,

                 dense3_num_units=200,
           #      dropout4_p=0.25,

                 dense4_num_units = 100, 
                 output_num_units=38,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.001,
                 update_momentum=0.9,
                 
                 eval_size=0.1,
                 verbose=1,
                 max_epochs=1000) 

#del X_scaled['VisitNumberCount'] 
net0.fit(data,yy)




#%%
resultNet = net0.predict_proba(test)
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle walmart competetion/sample_submission.csv')
Id = submit['VisitNumber']
submit.iloc[:,1:] = resultNet
submit.iloc[:,0] = Id
submit.to_csv('/Users/weizhi/Desktop/kaggle walmart competetion/deeplearning.csv',index = False)





