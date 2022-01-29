# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:13:41 2022

@author: metri
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import mymetrics as mts
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier


# import Orange
# import pickle

#%%
################################ Gini preprocessing model ###############################
# import data and split
df = pd.read_csv('results_mayonnaise_2_NORM.csv')
X = np.asarray(df.loc[:,[ 'Sdgreen',
                     'minred',
                     'Sdsat',
                     'SdL',
                     'maxa',
                     'skewness_b',
                     'corr',
                     'meas_corr_2',
                                ]]) #metrics
y = np.asarray(df.loc[:,"Grade"])
                    




my_KNN = KNeighborsClassifier(n_neighbors=11, #6 
                     weights='distance', algorithm='kd_tree', 
                     leaf_size=30, p=2, metric='euclidean', 
                     metric_params=None, n_jobs=None)

metrics = {"accuracy": [],
           "precision": [],
           "recall": [],
           "f1": []
    }
for key in metrics:
    print("kfold cross validation for {}".format(key))
    score = str(key)
    result = cross_validate(my_KNN, X, y, cv=10, scoring = score, return_estimator=True)
    metrics[key].append(result["test_score"])
    metrics[key].append(result["test_score"].mean())
    metrics[key].append(result["test_score"].std())
    
    
#%%%

############################### InfoGain Value > 0.2 ##############################
# import data and split
df = pd.read_csv('results_mayonnaise_2_NORM.csv')
X2 = np.asarray(df.loc[:,['Xsat',
                        'Sdsat',
                        'Sdb',
                        'diff_gray',
                        'maxb',
                        'kurtosis_b',
                        'Sdblue',
                        'maxsat',
                        'minsat',
                        'Xb',
                        'minblue',
                        'kurtosis_hue',
                        'entropy_diff',
                        'skewness_hue',
                        'homogeneity',
                        'dissimilarity',
                        'inv_diff',
                        'kurtosis_saturation',
                        'Xa',
                        'meas_corr_2',
                                    ]]) #metrics
y2 = np.asarray(df.loc[:,"Grade"])
                    
my_KNN_2 = KNeighborsClassifier(n_neighbors=6, #4 
                     weights='distance', algorithm='kd_tree', 
                     leaf_size=10, p=2, metric='euclidean', 
                     metric_params=None, n_jobs=None)

metrics2 = {"accuracy": [],
           "precision": [],
           "recall": [],
           "f1": []
    }
for key in metrics2:
    print("kfold cross validation for {}".format(key))
    score = str(key)
    result = cross_validate(my_KNN_2, X2, y2, cv=10, scoring = score, return_estimator=True)
    metrics2[key].append(result["test_score"])
    metrics2[key].append(result["test_score"].mean())
    metrics2[key].append(result["test_score"].std())
