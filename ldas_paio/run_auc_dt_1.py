#サンプリング後のオーバーラッピングデータの割合を調べない
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
from ldas_paio_2 import LDAS_PAIO_2
from ldas_paio_3 import LDAS_PAIO_3
from ldas_paio_4 import LDAS_PAIO_4
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
    print("データセット名  IR  提案手法A  提案手法I  提案手法B  提案手法T  PAIO  LDAS  SMOTE")
    a=glob.glob('*.csv')
    all=[]
    rank1=[]

    for line in a:
        path = r'{}'.format(line)
        data_name=os.path.splitext(path)[0]
        X, y, Maj_num, Min_num, IR, features = load_norm_data(path)
        r=10
        k=5
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        all_auc=[]
        
        
        #-----提案手法A-----
        
        
        
        ave_auc=[]
        
        for i in range(r):
            auc_value=[]
            
            
            
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                X_re, y_re = LDAS_PAIO_1().fit_sample(X[train_index], y[train_index])
                
                
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
                
            ave_auc.append(sum(auc_value)/len(auc_value))
            
        ave=sum(ave_auc)/len(ave_auc)
        
        
        
        
        
        all_auc.append(ave)
        
        #-----提案手法I-----
        
        
        
        ave_auc=[]
        
        for i in range(r):
            auc_value=[]
            
            
            
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                X_re, y_re = LDAS_PAIO_2().fit_sample(X[train_index], y[train_index])
                
                
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
                
            ave_auc.append(sum(auc_value)/len(auc_value))
            
        ave=sum(ave_auc)/len(ave_auc)
        
        
        
        
        
        all_auc.append(ave)
        
        #-----提案手法B-----
        
        
        
        ave_auc=[]
        
        for i in range(r):
            auc_value=[]
            
            
            
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                X_re, y_re = LDAS_PAIO_3().fit_sample(X[train_index], y[train_index])
                
                
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
                
            ave_auc.append(sum(auc_value)/len(auc_value))
            
        ave=sum(ave_auc)/len(ave_auc)
        
        
        
        
        
        all_auc.append(ave)
        
        #-----提案手法T-----
        
        
        
        ave_auc=[]
        
        for i in range(r):
            auc_value=[]
            
            
            
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                X_re, y_re = LDAS_PAIO_4().fit_sample(X[train_index], y[train_index])
                
                
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
                
            ave_auc.append(sum(auc_value)/len(auc_value))
            
        ave=sum(ave_auc)/len(ave_auc)
        
        
        
        
        
        all_auc.append(ave)
        

        #-----PAIO-----
        
        
        
        ave_auc=[]
        
        for i in range(r):
            auc_value=[]
            
            
            
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                X_re, y_re = PAIO().fit_sample(X[train_index], y[train_index])
                
                
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
                
            ave_auc.append(sum(auc_value)/len(auc_value))
            
        ave=sum(ave_auc)/len(ave_auc)
        
        
        
        
        
        all_auc.append(ave)
        
        #-----LDAS-----
        
        
        
        ave_auc=[]
        
        for i in range(r):
            auc_value=[]
            
            
            
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                X_re, y_re = LDAS().fit_sample(X[train_index], y[train_index])
               
                
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
                
            ave_auc.append(sum(auc_value)/len(auc_value))
            
        ave=sum(ave_auc)/len(ave_auc)
        
        
        
        
        
        all_auc.append(ave)
        
        #-----SMOTE-----
        
        
        
        ave_auc=[]
        
        for i in range(r):
            auc_value=[]
            
            
            
            for train_index, test_index in skf.split(X, y):
                test_y=y[test_index]
                test_x=X[test_index]
                sm = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=71)
                X_re, y_re = sm.fit_resample(X[train_index], y[train_index])
                
                
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
                
            ave_auc.append(sum(auc_value)/len(auc_value))

        ave=sum(ave_auc)/len(ave_auc)
        
        
        
        
        
        all_auc.append(ave)
        
        print()
        print("{}  {}  {}  {}  {}  {}  {}  {}  {}\n".format(data_name, IR, all_auc[0], all_auc[1], all_auc[2], all_auc[3], all_auc[4], all_auc[5], all_auc[6]))
        
        all.append(all_auc)
        
        rank1.append((pd.Series(all_auc)).rank(ascending=False,method='min'))
    dataset_num=len(all)
    print()
    print("平均値  /  {}  {}  {}  {}  {}  {}  {}".format(sum(row[0] for row in all)/dataset_num,sum(row[1] for row in all)/dataset_num,sum(row[2] for row in all)/dataset_num,sum(row[3] for row in all)/dataset_num,sum(row[4] for row in all)/dataset_num,sum(row[5] for row in all)/dataset_num,sum(row[6] for row in all)/dataset_num))
    print("平均順位  /  {}  {}  {}  {}  {}  {}  {}\n".format(sum(row[0] for row in rank1)/dataset_num,sum(row[1] for row in rank1)/dataset_num,sum(row[2] for row in rank1)/dataset_num,sum(row[3] for row in rank1)/dataset_num,sum(row[4] for row in rank1)/dataset_num,sum(row[5] for row in rank1)/dataset_num,sum(row[6] for row in rank1)/dataset_num))
    