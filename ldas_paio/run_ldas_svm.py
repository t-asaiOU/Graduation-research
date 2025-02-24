import os
import re
import csv
import glob
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


if __name__ == '__main__':
    a=glob.glob('*.csv')
    for line in a:
        path = r'{}'.format(line)
        X, y, Maj_num, Min_num, IR, features = load_norm_data(path)
        i=0
        r=10
        k=5
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        ave_auc=[]
        for i in range(r):
            auc_value=[]
            for train_index, test_index in skf.split(X, y):
                X_re, y_re = LDAS().fit_sample(X[train_index], y[train_index])
                clf=svm.SVC(gamma="scale",probability=True)
                clf.fit(X_re,y_re)
                test_y=y[test_index]
                test_x=X[test_index]
                y_pred_prob=clf.predict_proba(test_x)[:,-1]
                auc_value.append(roc_auc_score(test_y,y_pred_prob))
                #print("AUC: {}".format(roc_auc_score(test_y,y_pred_prob)))
            ave_auc.append(sum(auc_value)/len(auc_value))
        ave=sum(ave_auc)/len(ave_auc)
        file_name=os.path.splitext(os.path.basename(__file__))[0]
        data_name=os.path.splitext(path)[0]
        print("データセット名: {}".format(data_name))
        print("手法名: {}".format(file_name.split('_',2)[1]))
        print("分類器名: {}".format(file_name.split('_',2)[2]))
        print("{}分割層化交差検証で計算されたAUCの{}回の平均: {}\n".format(k,r,ave))
