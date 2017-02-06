# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:31:35 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:57:13 2015

@author: Algorithm 001
"""


#%% loading each files
import glob, os
import collections
import pandas as pd

# https://docs.python.org/2/library/os.html
def findFilePath(path):
    os.chdir(path)
    filePaths = []
    for file in glob.glob("*.csv"):
        filePaths.append(file)
    return filePaths




#%% testing
path = '/Users/weizhi/Desktop/kaggle_Rossmann/result/'
filePaths = findFilePath(path)

result0 = pd.read_csv(filePaths[1])

result1 = pd.read_csv(filePaths[2])
result2 = pd.read_csv(filePaths[3])
result3 = pd.read_csv(filePaths[4])


result = pd.DataFrame()
result['Id']= result0['Id']

result['Sales'] = 0.25*result0['Sales'] + 0.25*result1['Sales'] + 0.25*result2['Sales']+ 0.25*result3['Sales']
result.to_csv('/Users/weizhi/Desktop/kaggle_Rossmann/result/result_Ensemble',index=False)




















