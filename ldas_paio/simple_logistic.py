import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

data=np.loadtxt('abalone19.csv',delimiter=',')
y=data[:,-1].astype(float)
colums=data.shape[1]
x=data[:,:colums-1]

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

logreg=LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
#print(y_pred)
y_pred_prob=logreg.predict_proba(X_test)[:,-1]
#print(y_pred_prob)

fpr,tpr,thresholds=roc_curve(y_test,y_pred_prob)
print("AUC: {}".format(roc_auc_score(y_test,y_pred_prob)))

cv_auc=cross_val_score(logreg,x,y,cv=5,scoring='roc_auc')
print("5回の交差検証で計算されたAUC: {}".format(cv_auc))