# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 11:36:48 2022

@author: metri
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import mymetrics as mts
from sklearn.model_selection import cross_validate


df = pd.read_csv('results_mayonnaise_2_NORM.csv')
X = np.asarray(df.loc[:,[ 'kurtosis_green',
                     'std_yellow',
                     'Xhue',
                     'Sdsat',
                     'kurtosis_saturation',
                     'skewness_b',
                     'kurtosis_L',
                     'corr'
                            ]]) #metrics
y = np.asarray(df.loc[:,"Grade"])


my_svc = SVC(C=10.5, kernel='rbf', gamma = 'auto', tol=0.001, cache_size=500, 
             verbose=True, max_iter=- 1, decision_function_shape='ovr',
             random_state=1)



metrics = {"accuracy": [],
           "precision": [],
           "recall": [],
           "f1": []
    }
for key in metrics:
    print("\nkfold cross validation for {}\n".format(key))
    score = str(key)
    result = cross_validate(my_svc, X, y, cv=10, scoring = score)
    metrics[key].append(result["test_score"])
    metrics[key].append(result["test_score"].mean())
    metrics[key].append(result["test_score"].std())

