import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold, LeaveOneOut
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn import svm
import pandas as pd
import math
SVM = svm.SVC
from sklearn.ensemble import RandomForestClassifier as RF, RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from print_data import print_data
import jackknife
import os
from creat_self_models import MaxEnt


def maxent_xlf(maxstep=10):
    return MaxEnt(maxstep=maxstep)


def ANN_xlf():
    return MLPClassifier(random_state=43)


def gbrt_xlf(learning_rate=0.1,n_estimators=100):
    return GradientBoostingClassifier(learning_rate=learning_rate,n_estimators=n_estimators,random_state=43)


def xgb_xlf(md=2, lr=0.01, nesti=1800):
    return xgb.XGBClassifier(max_depth=md,
                             learning_rate=lr,
                             n_estimators=nesti,
                             objective='binary:logistic',
                             nthread=-1,
                             gamma=0,
                             min_child_weight=1,
                             max_delta_step=0,
                             subsample=0.85,
                             colsample_bytree=0.7,
                             colsample_bylevel=1,
                             reg_alpha=0,
                             reg_lambda=1,
                             scale_pos_weight=1,
                             seed=1440,
                             missing=None,
                             n_jobs=-1)


def svm_xlf(svm_c=1, svm_g=2 ** -3):
    return svm.SVC(C=svm_c, kernel='rbf', gamma=svm_g, probability=True)


def rf_xlf(md=3, nesti=400):
    return RandomForestClassifier(max_depth=md, n_estimators=nesti, n_jobs=-1)


def knn_xlf(n=2, p=1):
    return KNeighborsClassifier(n_neighbors=n, p=p, weights="distance", algorithm="auto", n_jobs=-1)


param_grid = {
    'svm':
        {
            'class_weight': [{0: 1, 1: 1.1}, {0: 1, 1: 1}, {0: 1.1, 1: 1.0}],
            "kernel": ['rbf', 'sigmoid', 'linear'],
            "gamma": [x * 0.0015 for x in range(30, 60)],
            "C": [x * 0.1 for x in range(12, 20) if x % 2 == 0],
            'probability': [True],
        },
    'rf':
        {
            'max_depth': [x for x in range(2, 20) if x%2==0],
            'n_estimators': [x * 100 for x in range(10, 30) if x%2 ==0],
            # 'min_sample_leaf': [x * 10 for x in range(5, 6)],
            'min_weight_fraction_leaf': [x * 0.01 for x in range(0, 20) if x%2==0],
        },

    'knn': {
        'n_neighbors': [x * 2 for x in range(1, 10)],
        'p': [x for x in range(1, 10)],
    },
    'xgb': {
        'reg_alpha': [x * 0.001 for x in range(0, 10)],
        'max_depth': [x * 2 for x in range(1, 10)],
        'learning_rate': [x * 0.0008 for x in range(1,101) if x % 10 == 0],
        'n_estimators': [x * 100 for x in range(10, 30) if x%2==0],

    },

    'maxnet': {
        'maxstep': [10,100,200],
    },

    'gbrt': {
        'learning_rate': [0.1, 0.01, 0.05],
        'n_estimators': [100, 200, 400],

    },

}


xlf_dict = {'ann':ANN_xlf(),'rf':rf_xlf(),'maxnet':maxent_xlf(),'gbrt':gbrt_xlf(),'xgb':xgb_xlf()}

