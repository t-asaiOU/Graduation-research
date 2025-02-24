import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


data=np.loadtxt('out_train.csv',delimiter=',')
y=data[:,-1].astype(float)
colums=data.shape[1]
x=data[:,:colums-1]

clf=svm.SVC(gamma="scale",probability=True)
clf.fit(x,y)
data_test=np.loadtxt('out_test.csv',delimiter=',')
test_y=data_test[:,-1].astype(float)
test_x=data_test[:,:colums-1]

y_pred=clf.predict(test_x)
#print(y_pred)
y_pred_prob=clf.predict_proba(test_x)[:,-1]
#print(y_pred_prob)


fpr,tpr,thresholds=roc_curve(test_y,y_pred_prob)

print("AUC: {}".format(roc_auc_score(test_y,y_pred_prob)))


cv_auc=cross_val_score(clf,x,y,cv=5,scoring='roc_auc')
print("5回の交差検証で計算されたAUC: {}".format(cv_auc))


kf = KFold(5, shuffle=True)
for i in range(10)
	for train_index, test_index in kf.split(x):
		train_x = x[train_index]
		train_y = y[train_index]
		test_x = x[test_index]
		test_y = y[test_index]