from os import path
from dataPreprocessing import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os

def _get_image_names(base_path, imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': path.join(base_path,'three_band/{}.tif'.format(imageId)),             # (3, 3348, 3403)
         'A': path.join(base_path,'sixteen_band/{}_A.tif'.format(imageId)),         # (8, 134, 137)
         'M': path.join(base_path,'sixteen_band/{}_M.tif'.format(imageId)),         # (8, 837, 851)
         'P': path.join(base_path,'sixteen_band/{}_P.tif'.format(imageId)),         # (3348, 3403)
         }
    return d


inDir = '/Users/weizhi/Desktop/kaggle_code_combine/three_bands_kaggle' #'/home/n01z3/dataset/dstl'

# read the training data from train_wkt_v4.csv
df = pd.read_csv(inDir + '/train_wkt_v4.csv')

# grid size will also be needed later..
gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


N_Cls = 10
DF = pd.read_csv(inDir + '/train_wkt_v4.csv')
GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(os.path.join(inDir, 'sample_submission-3.csv'))
ISZ = 160
smooth = 1e-12

DF.ImageId.unique()
SB.ImageId.unique()
#
#
# image = _get_image_names(inDir, '6120_2_2')

ID = []
for item in DF.ImageId.unique():
    ID.append(item.split('_')[0])
print set(ID)



ID_test = []
for item in SB.ImageId.unique():
    ID_test.append(item.split('_')[0])
print set(ID_test)



image_dict = {}
image_path = _get_image_names(inDir, "6120_2_2")
for path_key in image_path.keys():
    image_path_read = image_path[path_key]
    print image_path_read
    path_image_item = tiff.imread(image_path_read)
    print path_image_item.shape
    image_dict[path_key] = path_image_item

#for i in range(10)

df_train = {}#pd.DataFrame( columns=range(1,11))
df_train_value = {}
count = 0
train_image = []
test_image = []
for i in range(5):
    name_image = "6120_2_" + str(i)

    image_path = _get_image_names(inDir, name_image)
    if name_image in SB.ImageId.unique():
        test_image.append(name_image)
    if name_image in DF.ImageId.unique():
        train_image.append(name_image)
        image_pixel_value = tiff.imread(image_path['3']) # get the three bands image channels
        image_size = image_pixel_value.shape[1:] # get the image size
        for classID in range(1,11):
            mask = generate_mask_for_image_and_class(image_size, "6120_2_" + str(i), classID, gs, df)
            x, y = np.where(mask==1)
            if classID not in df_train.keys():
                df_train[classID] = image_pixel_value[:,x,y]
            else:
                df_train[classID] = np.concatenate((df_train[classID], image_pixel_value[:,x,y]), axis=1)


data_train = pd.DataFrame(columns = ['label'] + [str(x) for x in range(1,11)])
count = 0



num_train =  df_train[1].shape[1]
num_index = np.random.choice(num_train, 10000)
data_get = df_train[1][:,num_index]
label_data = np.ones(10000)*1
X = np.concatenate((label_data.reshape(1,10000), data_get), axis = 0)
X = np.transpose(X)

for i in range(2,11):
    print "the %d class" %i
    num_train = df_train[i].shape[1]
    if num_train>0:
        size_data = 0
        if num_train>=10000:
            num_index = np.random.choice(num_train, 10000)
            data_get = df_train[i][:,num_index]
            size_data = 10000

        else:
            num_index = np.random.choice(num_train, num_train)
            data_get = df_train[i]
            size_data = num_train

        label_data = np.ones(size_data)*i
        X_1 = np.concatenate((label_data.reshape(1, size_data), data_get), axis = 0)
        X_1 = np.transpose(X_1)
        X = np.concatenate((X, X_1), axis = 0)

# train a model

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

X_2= X
np.random.shuffle(X)

y = X[:,0]
X_data = X[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.8,random_state=42)


# Logistic regression with no calibration as baseline
lr = LogisticRegression(C=1., solver='lbfgs', multi_class='multinomial', max_iter= 1000, class_weight= 'balanced')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

clf = LogisticRegression(C=1., solver='lbfgs', multi_class='multinomial', max_iter= 1000, class_weight= 'balanced')
clf.fit(X_data, y)


# test data

image_path = _get_image_names(inDir, '6120_2_1')

image_pixel_value = tiff.imread(image_path['3']) # get the three bands image channels


mask_shape = image_pixel_value.shape[1:]

predict_image = np.reshape(image_pixel_value, (mask_shape[1]*mask_shape[0], 3))

predict_label = clf.predict(predict_image)

predict_mask = np.reshape(predict_label, (mask_shape[0], mask_shape[1]))

plt.imshow(predict_mask)







