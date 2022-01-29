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


#%%

########################### Info gain preprocessing model ################################

# import and split data
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


#y_pred = my_svc.predict(X_test)
# Metrics
#svc_metrics = mts.create_metrics(y_test, y_pred)
#svc_cmat = confusion_matrix(y_test, y_pred, labels=[0,1])
#report = classification_report(y_test, y_pred, output_dict=True)

#%%

################################# Info Gain > 0.2 ################################

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

my_svc_2 = SVC(C=9.5, kernel='rbf', gamma = 'auto', tol=0.001, cache_size=500, 
             verbose=True, max_iter=- 1, decision_function_shape='ovr',
             random_state=1)


metrics2 = {"accuracy": [],
           "precision": [],
           "recall": [],
           "f1": []
    }
for key in metrics2:
    print("kfold cross validation for {}".format(key))
    score = str(key)
    result = cross_validate(my_svc_2, X2, y2, cv=10, scoring = score, n_jobs = 3)
    metrics2[key].append(result["test_score"])
    metrics2[key].append(result["test_score"].mean())
    metrics2[key].append(result["test_score"].std())
