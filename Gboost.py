# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 23:59:57 2022

@author: metri
"""

### GradientBoosting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import mymetrics as mts
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer


df = pd.read_csv('results_mayonnaise_2_NORM.csv')
X = np.asarray(df.loc[:,['Sdsat',
                        'minhue',
                        'maxL',
                        'maxa',
                        'skewness_b',
                        'kurtosis_L',
                        'meas_corr_2',
                            ]]) #metrics
y = np.asarray(df.loc[:,"Grade"])


my_GB = GradientBoostingClassifier(criterion='friedman_mse',
                                   loss='exponential',
                                   learning_rate=0.10,
                                   n_estimators=100,
                                   min_samples_split=5,#7
                                   min_samples_leaf=7, #7
                                   random_state=1,
                                   verbose=2,
                                   max_depth=3, #3
                                   warm_start = True,
                                   n_iter_no_change=20)

metrics = {"accuracy": [],
           "precision": [],
           "recall": [],
           "f1": []
    }
for key in metrics:
    print("kfold cross validation for {}".format(key))
    score = str(key)
    result = cross_validate(my_GB, X, y, cv=10, scoring = score, return_estimator=True)
    metrics[key].append(result["test_score"])
    metrics[key].append(result["test_score"].mean())
    metrics[key].append(result["test_score"].std())

