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
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


if __name__ == '__main__':
    a=glob.glob('*.csv')
    b=0
    j=0
    for line in a:
        j=j+1
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
                test_y=y[test_index]
                test_x=X[test_index]
                model = DecisionTreeClassifier()
                alpha = model.cost_complexity_pruning_path(X_re, y_re)
                ccp_alphas, impurities = alpha.ccp_alphas, alpha.impurities
                #print("ccp_alphas: {}".format(ccp_alphas))
                models = []
                for ccp_alpha in ccp_alphas:
                    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
                    model.fit(X_re, y_re)
                    models.append(model)
                test_scores = [model.score(test_x, test_y) for model in models[:-1]]
                #print("test_scores: {}".format(test_scores))
                max_score=max(test_scores)
                max_index=test_scores.index(max_score)
                #print("max_index: {}".format(max_index))
                #print("ccp_alphas[max_index]: {}".format(ccp_alphas[max_index]))
                model = DecisionTreeClassifier(ccp_alpha=ccp_alphas[max_index])
                model.fit(X_re, y_re)
                y_pred_prob=model.predict_proba(test_x)[:,-1]
                #print(model.predict_proba(test_x))
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
        b=b+ave
    print("AUCの平均: {}".format(b/j))