import cv2
import os
import base64
import requests
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

tranmodel = pickle.load(open('carTranModel.pkl','rb'))
tranmodel_np = np.array(tranmodel)
X_train = tranmodel_np[:,0:-1]
y_train = tranmodel_np[:,-1]
# print(X_train)
# print(y_train)

testmodel = pickle.load(open('carTestModel.pkl','rb'))
testmodel_np = np.array(testmodel)
X_test = testmodel_np[:,0:-1]
y_test = testmodel_np[:,-1]

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

path_model = 'D:\\year3term1\\aiAssignment1\\tranModel\\carModel.pkl'
pickle.dump(clf,open(path_model,'wb'))
print("car model is done...")



