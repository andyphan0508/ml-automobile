# import
from pandas.core.groupby.generic import DataFrameGroupBy
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn import svm
from sklearn import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name = "D:/Study/Machine Learning/automobile/imports-85.data"
dataframe = pd.read_csv (file_name)

print ("Shape of Automobile data: ",dataframe.shape)
print ("Size of Automobile data: " ,dataframe.size)
print (dataframe.head(10)) # <-- First 5 data set.

# Training Session
dataframe = dataframe.sample (n = 100)
dataframe.tail (3)
print (len(dataframe))
a = dataframe.values
y = a[:, -1:]
print (y.shape)
print (a.shape)
print (y[:5])
print (y.transpose())

# Encoding data to 0 and 1
LE = LabelEncoder()
data = LE.fit_transform(y)
print ("Label Encoder\n",data[:5])
#y1 = y[y == 1]
#y2 = y[y == 0]
#print (len(y1))
#print (len(y2))

b = dataframe.select_dtypes (include = ['float64', 'int64'])
print (b)
X = b.values
print ("X values\n",X[5])


# Scale to proper data set
data = b
scaler = MinMaxScaler ()
scaler.fit (data)
X = scaler.transform(data)
print ("\nScaled data\n",X[:5])

# --Training--
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.01, random_state = 3)
print ('\nTraining data : ', X_train.shape, y_train.shape)
print ('Testing data    : ', X_test.shape, y_test.shape)

# Prediction:
svm_classifier = svm.SVC (kernel = 'linear', gamma = 0.1, C = 3)
svm_classifier.fit(X_train, y_train)

y_pred_svm = svm_classifier.predict(X_test)
print ('Prediction value are:',y_pred_svm)
print ('True value are      :', y_test)

model = LogisticRegression() # <-- Phương thức train: Regression
model = model.fit(X_train, y_train)
y_pred = model.predict (X_test)
print (y_pred_svm)
accu_score = metrics.accuracy_score(y_pred_svm, y_test)
print('Accuracy: ',accu_score)
y_probability = model.predict_proba(X_test)
print('Probability\n', y_probability[-3:])