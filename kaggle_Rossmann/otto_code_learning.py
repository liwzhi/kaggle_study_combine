# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:21:59 2015

@author: weizhi
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet
from scipy.special import expit
import random
from sklearn.neural_network import BernoulliRBM
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding, MDS
from sklearn.decomposition import TruncatedSVD
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.svm import LinearSVC, SVC
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier

random.seed(21)
np.random.seed(21)

LINES = 61877

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(np.log(1+X))
    rbm1 = SVC(C=100.0, gamma = 0.1, probability=True, verbose=1).fit(X[0:9999,:], y[0:9999])
    rbm2 = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features='auto', bootstrap=False, oob_score=False, n_jobs=1, verbose=1).fit(X[0:9999,:], y[0:9999])
    rbm3 = GradientBoostingClassifier(n_estimators=50,max_depth=11,subsample=0.8,min_samples_leaf=5,verbose=1).fit(X[0:9999,:], y[0:9999])
    X =  np.append(X[10000:LINES,:], np.power(rbm1.predict_proba(X[10000:LINES,:])*rbm2.predict_proba(X[10000:LINES,:])*rbm3.predict_proba(X[10000:LINES,:]), (1/3.0))   , 1)
    return X, y[10000:LINES], encoder, scaler, rbm1, rbm2, rbm3

def load_test_data(path, scaler, rbm1, rbm2, rbm3):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(np.log(1+X))
    X =  np.append(X, np.power(rbm1.predict_proba(X)*rbm2.predict_proba(X)*rbm3.predict_proba(X), (1/3.0)), 1)
    return X, ids

def make_submission(y_prob, ids, encoder, name='/home/mikeskim/Desktop/kaggle/otto/data/lasagneSeed21.csv'):
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))


#Load Data
X, y, encoder, scaler, rbm1, rbm2, rbm3 = load_train_data('/home/mikeskim/Desktop/kaggle/otto/data/train.csv')
X_test, ids = load_test_data('/home/mikeskim/Desktop/kaggle/otto/data/test.csv', scaler, rbm1, rbm2, rbm3)

num_classes = len(encoder.classes_)
num_features = X.shape[1]

print(num_classes); print(num_features); print(X)


layers0 = [('input', InputLayer),
('dropoutf', DropoutLayer),
('dense0', DenseLayer),
('dropout', DropoutLayer),
('dense1', DenseLayer),
('dropout2', DropoutLayer),
('dense2', DenseLayer),
('output', DenseLayer)]


net0 = NeuralNet(layers=layers0,

input_shape=(None, num_features),
dropoutf_p=0.1,
dense0_num_units=600,
dropout_p=0.3,
dense1_num_units=600,
dropout2_p=0.1,
dense2_num_units=600,

output_num_units=num_classes,
output_nonlinearity=softmax,

#update=nesterov_momentum,
update=adagrad,
update_learning_rate=0.008,
eval_size=0.2,
verbose=1,
max_epochs=20)



net0.fit(X, y)
y_prob = net0.predict_proba(X_test)
num_runs = 50

for jj in xrange(num_runs):
  print(jj)
  X, y, encoder, scaler, rbm1, rbm2, rbm3 = load_train_data('/home/mikeskim/Desktop/kaggle/otto/data/train.csv')
  X_test, ids = load_test_data('/home/mikeskim/Desktop/kaggle/otto/data/test.csv', scaler, rbm1, rbm2, rbm3)
  num_classes = len(encoder.classes_)
  num_features = X.shape[1]
  net0.fit(X, y)
  y_prob = y_prob + net0.predict_proba(X_test)


y_prob = y_prob/(num_runs+1.0)
make_submission(y_prob, ids, encoder)








require(xgboost)
require(methods)
require(randomForest)
library(Rtsne)
require(data.table)
options(scipen=999)
set.seed(1004)

#41599 public

train = fread('/home/mikeskim/Desktop/kaggle/otto/data/train.csv',header=TRUE,data.table=F)
test = fread('/home/mikeskim/Desktop/kaggle/otto/data/test.csv',header=TRUE,data.table = F)
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))

x = 1/(1+exp(-sqrt(x)))

tsne <- Rtsne(as.matrix(x), check_duplicates = FALSE, pca = FALSE, 
              perplexity=30, theta=0.5, dims=2)

x = round(x,3)

##test here
#trind = 1:length(y)
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 7)

#bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
#                nfold = 4, 
#                column_subsample = 0.8, #subsample changes only hurt.
#                nrounds=60, max.depth=11, eta=0.46, min_child_weight=10)
                
#[59]  train-mlogloss:0.204678+0.004478	test-mlogloss:0.488262+0.011556 no add
#[55]  train-mlogloss:0.190645+0.003466	test-mlogloss:0.479680+0.010798 added tsne 


x = cbind(x, tsne$Y[,1]) 
x = cbind(x, tsne$Y[,2])

trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

trainX = x[trind,]
testX = x[teind,]


# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 7)


# Train the model
nround = 1200
bst = xgboost(param=param, data = trainX, label = y,
              nrounds=nround, max.depth=8, eta=0.03, min_child_weight=3)

# Make prediction
pred = predict(bst,testX)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)


tmpC = 1:240
tmpL = length(trind)
gtree = 200
for (z in tmpC) {
  print(z)
  tmpS1 = sample(trind,size=tmpL,replace=T)
  tmpS2 = setdiff(trind,tmpS1)
  
  tmpX2 = trainX[tmpS2,]
  tmpY2 = y[tmpS2]
  
  cst = randomForest(x=tmpX2, y=as.factor(tmpY2), replace=F, ntree=100, do.trace=T, mtry=7)
  
  tmpX1 = trainX[tmpS1,]
  tmpY1 = y[tmpS1]
  
  tmpX2 = predict(cst, tmpX1, type="prob")
  tmpX3 = predict(cst, testX, type="prob")
  
  bst = xgboost(param=param, data = cbind(tmpX1,tmpX2), label = tmpY1, column_subsample = 0.8, 
                nrounds=60, max.depth=11, eta=0.46, min_child_weight=10) 
  
  # Make prediction
  pred0 = predict(bst,cbind(testX,tmpX3))
  pred0 = matrix(pred0,9,length(pred0)/9)
  pred = pred + t(pred0)
}
pred = pred/(z+1)

pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='/home/mikeskim/Desktop/kaggle/otto/data/ottoHomeBagG4.csv', quote=FALSE,row.names=FALSE)


