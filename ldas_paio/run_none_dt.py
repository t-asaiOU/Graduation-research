import os
import re
import csv
import numpy as np
import pandas as pd
from ldas import LDAS
from utils import load_norm_data
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import tree
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN


if __name__ == '__main__':
    path = r'car-vgood.csv'
    X, y, Maj_num, Min_num, IR, features = load_norm_data(path)
    skf = StratifiedKFold(5, shuffle=True)
    i=0
    ave_auc=[]
    for i in range(10):
        auc_value=[]
        for train_index, test_index in skf.split(X, y):
            model = tree.DecisionTreeClassifier(max_depth=2, random_state=1)
            model.fit(X[train_index],y[train_index])
            test_y=y[test_index]
            test_x=X[test_index]
        
            model.predict(test_x)
            y_pred_prob=model.predict_proba(test_x)[:,-1]

            auc_value.append(roc_auc_score(test_y,y_pred_prob))
            #print("AUC: {}".format(roc_auc_score(test_y,y_pred_prob)))
        ave_auc.append(sum(auc_value)/len(auc_value))
    ave=sum(ave_auc)/len(ave_auc)
    file_name=os.path.splitext(os.path.basename(__file__))[0]
    data_name=os.path.splitext(path)[0]
    print("データセット名: {}".format(data_name))
    print("手法名: {}".format(file_name.split('_',2)[1]))
    print("分類器名: {}".format(file_name.split('_',2)[2]))
    print("5分割層化交差検証で計算されたAUCの10回の平均: {}".format(ave))


