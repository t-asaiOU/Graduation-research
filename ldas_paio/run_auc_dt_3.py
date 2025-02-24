#サンプリング後のオーバーラッピングデータの割合を調べる
import os
import re
import csv
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from ldas import LDAS
from paio import PAIO
from ldas_paio_1 import LDAS_PAIO_1
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
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN


if __name__ == '__main__':
    file_name=os.path.splitext(os.path.basename(__file__))[0]
    print("手法名: 提案手法1, PAIO, LDAS, SMOTE, B-SMOTE, ADASYN")
    a=glob.glob('*.csv')
    all=[]
    rank1=[]
    overlap_list=[]
    for line in a:
        path = r'{}'.format(line)
        data_name=os.path.splitext(path)[0]
        X, y, Maj_num, Min_num, IR, features = load_norm_data(path)
        r=10
        k=5
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        all_auc=[]
        all_overlap=[]
        z=0
        #-----提案手法1-----
        b=0
        c=0
        j=0
        ave_auc=[]
        ave_overlap=[]
        for i in range(r):
            auc_value=[]
            overlap_value=[]
            #z=z+1
            #print("{} ".format(z))
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                X_re, y_re = LDAS_PAIO_1().fit_sample(X[train_index], y[train_index])
                ratio=LDAS_PAIO_1().overlap_ratio(X_re, y_re)
                overlap_value.append(ratio)
                model = DecisionTreeClassifier()
                alpha = model.cost_complexity_pruning_path(X_re, y_re)
                ccp_alphas, impurities = alpha.ccp_alphas, alpha.impurities
                models = []
                for ccp_alpha in ccp_alphas:
                    if ccp_alpha<0:
                        continue
                    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
                    model.fit(X_re, y_re)
                    models.append(model)
                test_scores = [model.score(test_x, test_y) for model in models[:-1]]
                max_score=max(test_scores)
                max_index=test_scores.index(max_score)
                model = DecisionTreeClassifier(ccp_alpha=ccp_alphas[max_index])
                model.fit(X_re, y_re)
                y_pred_prob=model.predict_proba(test_x)[:,-1]
                auc_value.append(roc_auc_score(test_y,y_pred_prob))
                #print("AUC: {}".format(roc_auc_score(test_y,y_pred_prob)))
            ave_auc.append(sum(auc_value)/len(auc_value))
            ave_overlap.append(sum(overlap_value)/len(overlap_value))
        ave=sum(ave_auc)/len(ave_auc)
        ave1=sum(ave_overlap)/len(ave_overlap)
        #print("{}分割層化交差検証で計算されたAUCの{}回の平均: {}\n".format(k,r,ave))
        b=b+ave
        c=c+ave1
        j=j+1
        all_auc.append(b/j)
        all_overlap.append(c/j)
        #-----PAIO-----
        b=0
        c=0
        j=0
        ave_auc=[]
        ave_overlap=[]
        for i in range(r):
            auc_value=[]
            overlap_value=[]
            #z=z+1
            #print("{} ".format(z))
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                X_re, y_re = PAIO().fit_sample(X[train_index], y[train_index])
                ratio=LDAS_PAIO_1().overlap_ratio(X_re, y_re)
                overlap_value.append(ratio)
                model = DecisionTreeClassifier()
                alpha = model.cost_complexity_pruning_path(X_re, y_re)
                ccp_alphas, impurities = alpha.ccp_alphas, alpha.impurities
                models = []
                for ccp_alpha in ccp_alphas:
                    if ccp_alpha<0:
                        continue
                    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
                    model.fit(X_re, y_re)
                    models.append(model)
                test_scores = [model.score(test_x, test_y) for model in models[:-1]]
                max_score=max(test_scores)
                max_index=test_scores.index(max_score)
                model = DecisionTreeClassifier(ccp_alpha=ccp_alphas[max_index])
                model.fit(X_re, y_re)
                y_pred_prob=model.predict_proba(test_x)[:,-1]
                auc_value.append(roc_auc_score(test_y,y_pred_prob))
                #print("AUC: {}".format(roc_auc_score(test_y,y_pred_prob)))
            ave_auc.append(sum(auc_value)/len(auc_value))
            ave_overlap.append(sum(overlap_value)/len(overlap_value))
        ave=sum(ave_auc)/len(ave_auc)
        ave1=sum(ave_overlap)/len(ave_overlap)
        #print("{}分割層化交差検証で計算されたAUCの{}回の平均: {}\n".format(k,r,ave))
        b=b+ave
        c=c+ave1
        j=j+1
        all_auc.append(b/j)
        all_overlap.append(c/j)
        #-----LDAS-----
        b=0
        c=0
        j=0
        ave_auc=[]
        ave_overlap=[]
        for i in range(r):
            auc_value=[]
            overlap_value=[]
            #z=z+1
            #print("{} ".format(z))
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                X_re, y_re = LDAS().fit_sample(X[train_index], y[train_index])
                ratio=LDAS_PAIO_1().overlap_ratio(X_re, y_re)
                overlap_value.append(ratio)
                model = DecisionTreeClassifier()
                alpha = model.cost_complexity_pruning_path(X_re, y_re)
                ccp_alphas, impurities = alpha.ccp_alphas, alpha.impurities
                models = []
                for ccp_alpha in ccp_alphas:
                    if ccp_alpha<0:
                        continue
                    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
                    model.fit(X_re, y_re)
                    models.append(model)
                test_scores = [model.score(test_x, test_y) for model in models[:-1]]
                max_score=max(test_scores)
                max_index=test_scores.index(max_score)
                model = DecisionTreeClassifier(ccp_alpha=ccp_alphas[max_index])
                model.fit(X_re, y_re)
                y_pred_prob=model.predict_proba(test_x)[:,-1]
                auc_value.append(roc_auc_score(test_y,y_pred_prob))
                #print("AUC: {}".format(roc_auc_score(test_y,y_pred_prob)))
            ave_auc.append(sum(auc_value)/len(auc_value))
            ave_overlap.append(sum(overlap_value)/len(overlap_value))
        ave=sum(ave_auc)/len(ave_auc)
        ave1=sum(ave_overlap)/len(ave_overlap)
        #print("{}分割層化交差検証で計算されたAUCの{}回の平均: {}\n".format(k,r,ave))
        b=b+ave
        c=c+ave1
        j=j+1
        all_auc.append(b/j)
        all_overlap.append(c/j)
        #-----SMOTE-----
        b=0
        c=0
        j=0
        ave_auc=[]
        ave_overlap=[]
        for i in range(r):
            auc_value=[]
            overlap_value=[]
            #z=z+1
            #print("{} ".format(z))
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                sm = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=71)
                X_re, y_re = sm.fit_resample(X[train_index], y[train_index])
                ratio=LDAS_PAIO_1().overlap_ratio(X_re, y_re)
                overlap_value.append(ratio)
                model = DecisionTreeClassifier()
                alpha = model.cost_complexity_pruning_path(X_re, y_re)
                ccp_alphas, impurities = alpha.ccp_alphas, alpha.impurities
                models = []
                for ccp_alpha in ccp_alphas:
                    if ccp_alpha<0:
                        continue
                    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
                    model.fit(X_re, y_re)
                    models.append(model)
                test_scores = [model.score(test_x, test_y) for model in models[:-1]]
                max_score=max(test_scores)
                max_index=test_scores.index(max_score)
                model = DecisionTreeClassifier(ccp_alpha=ccp_alphas[max_index])
                model.fit(X_re, y_re)
                y_pred_prob=model.predict_proba(test_x)[:,-1]
                auc_value.append(roc_auc_score(test_y,y_pred_prob))
                #print("AUC: {}".format(roc_auc_score(test_y,y_pred_prob)))
            ave_auc.append(sum(auc_value)/len(auc_value))
            ave_overlap.append(sum(overlap_value)/len(overlap_value))
        ave=sum(ave_auc)/len(ave_auc)
        ave1=sum(ave_overlap)/len(ave_overlap)
        #print("{}分割層化交差検証で計算されたAUCの{}回の平均: {}\n".format(k,r,ave))
        b=b+ave
        c=c+ave1
        j=j+1
        all_auc.append(b/j)
        all_overlap.append(c/j)
        #-----Borderlen-SMOTE-----
        b=0
        c=0
        j=0
        ave_auc=[]
        ave_overlap=[]
        for i in range(r):
            auc_value=[]
            overlap_value=[]
            #z=z+1
            #print("{} ".format(z))
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                sm = BorderlineSMOTE()
                X_re, y_re = sm.fit_resample(X[train_index], y[train_index])
                ratio=LDAS_PAIO_1().overlap_ratio(X_re, y_re)
                overlap_value.append(ratio)
                model = DecisionTreeClassifier()
                alpha = model.cost_complexity_pruning_path(X_re, y_re)
                ccp_alphas, impurities = alpha.ccp_alphas, alpha.impurities
                models = []
                for ccp_alpha in ccp_alphas:
                    if ccp_alpha<0:
                        continue
                    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
                    model.fit(X_re, y_re)
                    models.append(model)
                test_scores = [model.score(test_x, test_y) for model in models[:-1]]
                max_score=max(test_scores)
                max_index=test_scores.index(max_score)
                model = DecisionTreeClassifier(ccp_alpha=ccp_alphas[max_index])
                model.fit(X_re, y_re)
                y_pred_prob=model.predict_proba(test_x)[:,-1]
                auc_value.append(roc_auc_score(test_y,y_pred_prob))
                #print("AUC: {}".format(roc_auc_score(test_y,y_pred_prob)))
            ave_auc.append(sum(auc_value)/len(auc_value))
            ave_overlap.append(sum(overlap_value)/len(overlap_value))
        ave=sum(ave_auc)/len(ave_auc)
        ave1=sum(ave_overlap)/len(ave_overlap)
        #print("{}分割層化交差検証で計算されたAUCの{}回の平均: {}\n".format(k,r,ave))
        b=b+ave
        c=c+ave1
        j=j+1
        all_auc.append(b/j)
        all_overlap.append(c/j)
        #-----ADASYN-----
        b=0
        c=0
        j=0
        ave_auc=[]
        ave_overlap=[]
        for i in range(r):
            auc_value=[]
            overlap_value=[]
            #z=z+1
            #print("{} ".format(z))
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                ada = ADASYN()
                X_re, y_re = ada.fit_resample(X[train_index], y[train_index])
                ratio=LDAS_PAIO_1().overlap_ratio(X_re, y_re)
                overlap_value.append(ratio)
                model = DecisionTreeClassifier()
                alpha = model.cost_complexity_pruning_path(X_re, y_re)
                ccp_alphas, impurities = alpha.ccp_alphas, alpha.impurities
                models = []
                for ccp_alpha in ccp_alphas:
                    if ccp_alpha<0:
                        continue
                    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
                    model.fit(X_re, y_re)
                    models.append(model)
                test_scores = [model.score(test_x, test_y) for model in models[:-1]]
                max_score=max(test_scores)
                max_index=test_scores.index(max_score)
                model = DecisionTreeClassifier(ccp_alpha=ccp_alphas[max_index])
                model.fit(X_re, y_re)
                y_pred_prob=model.predict_proba(test_x)[:,-1]
                auc_value.append(roc_auc_score(test_y,y_pred_prob))
                #print("AUC: {}".format(roc_auc_score(test_y,y_pred_prob)))
            ave_auc.append(sum(auc_value)/len(auc_value))
            ave_overlap.append(sum(overlap_value)/len(overlap_value))
        ave=sum(ave_auc)/len(ave_auc)
        ave1=sum(ave_overlap)/len(ave_overlap)
        #print("{}分割層化交差検証で計算されたAUCの{}回の平均: {}".format(k,r,ave))
        b=b+ave
        c=c+ave1
        j=j+1
        all_auc.append(b/j)
        all_overlap.append(c/j)
        print()
        print("{}: {}, {}, {}, {}, {}, {}\n".format(data_name, all_auc[0], all_auc[1], all_auc[2], all_auc[3], all_auc[4], all_auc[5]))
        print("オーバーラップ率: {}, {}, {}, {}, {}, {}\n".format(all_overlap[0], all_overlap[1], all_overlap[2], all_overlap[3], all_overlap[4], all_overlap[-1]))
        all.append(all_auc)
        overlap_list.append(all_overlap)
        rank1.append((pd.Series(all_auc)).rank(ascending=False,method='min'))
    dataset_num=len(all)
    print()
    print("平均値: {}, {}, {}, {}, {}, {}".format(sum(row[0] for row in all)/dataset_num,sum(row[1] for row in all)/dataset_num,sum(row[2] for row in all)/dataset_num,sum(row[3] for row in all)/dataset_num,sum(row[4] for row in all)/dataset_num,sum(row[5] for row in all)/dataset_num))
    print("平均順位: {}, {}, {}, {}, {}, {}\n".format(sum(row[0] for row in rank1)/dataset_num,sum(row[1] for row in rank1)/dataset_num,sum(row[2] for row in rank1)/dataset_num,sum(row[3] for row in rank1)/dataset_num,sum(row[4] for row in rank1)/dataset_num,sum(row[5] for row in rank1)/dataset_num))
    print("オーバーラップ率: {}, {}, {}, {}, {}, {}".format(sum(row[0] for row in overlap_list)/dataset_num,sum(row[1] for row in overlap_list)/dataset_num,sum(row[2] for row in overlap_list)/dataset_num,sum(row[3] for row in overlap_list)/dataset_num,sum(row[4] for row in overlap_list)/dataset_num,sum(row[5] for row in overlap_list)/dataset_num))