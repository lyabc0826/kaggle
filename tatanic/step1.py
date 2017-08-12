#coding:utf-8

import numpy as np # linear algebra
import pandas as pd
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
combine = [train_df, test_df]
#print(train_df.columns.values)
#print(train_df.head())
train_tmp = None
test_tmp = None
for i in range(len(combine)):
    dataset = combine[i].drop("SibSp", axis=1)
    dataset = dataset.drop("PassengerId", axis=1)
    dataset = dataset.drop("Parch", axis=1)
    dataset = dataset.drop("Name", axis=1)
    dataset = dataset.drop("Sex", axis=1)
    dataset = dataset.drop("Ticket", axis=1)
    dataset = dataset.drop("Cabin", axis=1)
    dataset = dataset.drop("Embarked", axis=1)
    if i == 0:
        train_tmp = dataset
    else:
        test_tmp = dataset
x_train = train_tmp.drop("Survived", axis=1)
y_train = train_tmp['Survived']
x_test  = test_tmp.copy()
print(x_train.columns.values)
x_train['Age'] = x_train['Age'].fillna(30)
x_test['Age'] = x_test['Age'].fillna(30)
x_test['Fare'] = x_test['Fare'].fillna(35)
print(x_test.describe())

#print(x_train.shape, y_train.shape, x_test.shape)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pre = logreg.predict(x_test)
acc_log = round(logreg.score(x_train,y_train)*100, 2)
print(acc_log)
