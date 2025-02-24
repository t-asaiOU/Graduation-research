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
from ldas_paio_2_3 import LDAS_PAIO_2_3
from utils import load_norm_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


if __name__ == '__main__':
    file_name=os.path.splitext(os.path.basename(__file__))[0]

    print("データセット名  IR  提案手法A  提案手法I  提案手法B  提案手法T  提案手法I+B  PAIO  LDAS  SMOTE")
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
        r_auc=[]
    
        for i in range(r):
            k_auc=[]

            for train_index, test_index in skf.split(X, y):
                auc_value=[]
                test_y=y[test_index]
                test_x=X[test_index]

                #-----提案手法A-----

                X_re, y_re = LDAS_PAIO_1().fit_sample(X[train_index], y[train_index])
                auc_value.append(LDAS_PAIO_1().auc_dt(X_re, y_re, test_x, test_y))

                #-----提案手法I-----

                X_re, y_re = LDAS_PAIO_2().fit_sample(X[train_index], y[train_index])
                auc_value.append(LDAS_PAIO_1().auc_dt(X_re, y_re, test_x, test_y))

                #-----提案手法B-----

                X_re, y_re = LDAS_PAIO_3().fit_sample(X[train_index], y[train_index])
                auc_value.append(LDAS_PAIO_1().auc_dt(X_re, y_re, test_x, test_y))

                #-----提案手法T-----

                X_re, y_re = LDAS_PAIO_4().fit_sample(X[train_index], y[train_index])
                auc_value.append(LDAS_PAIO_1().auc_dt(X_re, y_re, test_x, test_y))

                #-----提案手法I+B-----

                X_re, y_re = LDAS_PAIO_2_3().fit_sample(X[train_index], y[train_index])
                auc_value.append(LDAS_PAIO_1().auc_dt(X_re, y_re, test_x, test_y))

                #-----PAIO-----

                X_re, y_re = PAIO().fit_sample(X[train_index], y[train_index])
                auc_value.append(LDAS_PAIO_1().auc_dt(X_re, y_re, test_x, test_y))

                #-----LDAS-----

                X_re, y_re = LDAS().fit_sample(X[train_index], y[train_index])
                auc_value.append(LDAS_PAIO_1().auc_dt(X_re, y_re, test_x, test_y))

                #-----SMOTE-----

                sm = SMOTE()
                X_re, y_re = sm.fit_resample(X[train_index], y[train_index])
                auc_value.append(LDAS_PAIO_1().auc_dt(X_re, y_re, test_x, test_y))

                k_auc.append(auc_value)   
        
            r_auc.append(k_auc)
        
        ave=np.mean(np.array(r_auc), axis=(0,1))
        ave=ave.tolist()

        print("{}  {}  {}  {}  {}  {}  {}  {}  {}  {}".format(data_name, IR, ave[0], ave[1], ave[2], ave[3], ave[4], ave[5], ave[6], ave[7]))
       
        all.append(ave)
        
        rank1.append((pd.Series(ave)).rank(ascending=False,method='min'))
    dataset_num=len(all)

    print("平均値  /  {}  {}  {}  {}  {}  {}  {}  {}".format(sum(row[0] for row in all)/dataset_num,sum(row[1] for row in all)/dataset_num,sum(row[2] for row in all)/dataset_num,sum(row[3] for row in all)/dataset_num,sum(row[4] for row in all)/dataset_num,sum(row[5] for row in all)/dataset_num,sum(row[6] for row in all)/dataset_num,sum(row[7] for row in all)/dataset_num))
    print("平均順位  /  {}  {}  {}  {}  {}  {}  {}  {}".format(sum(row[0] for row in rank1)/dataset_num,sum(row[1] for row in rank1)/dataset_num,sum(row[2] for row in rank1)/dataset_num,sum(row[3] for row in rank1)/dataset_num,sum(row[4] for row in rank1)/dataset_num,sum(row[5] for row in rank1)/dataset_num,sum(row[6] for row in rank1)/dataset_num,sum(row[7] for row in rank1)/dataset_num))
    