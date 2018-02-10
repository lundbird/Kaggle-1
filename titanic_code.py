# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:48:25 2017

@author: alex
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

#Import
train=pd.read_csv(r'C:\Users\alex\Desktop\Programming Projects\titanic\train.csv')
test=pd.read_csv(r'C:\Users\alex\Desktop\Programming Projects\titanic\test.csv')
t=pd.concat([train,test])

#EDA
#print(t.head() )
#print(t.info())
#print(t.columns )

#Feature Engineering
t.Cabin=t.Cabin.astype(str).str[0]

t.Name=t.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
t.Name=t.Name.replace(['Dr','Sir','Don','Rev','Master'],'Mr')
t.Name=t.Name.replace(['Miss','Mme','Ms','Lady','Mlle'], 'Mrs')
t.Name=t.Name.replace(['Major','Col','Capt','Countess','Jonhkeer'],'R')

t.Age=t.groupby(['Pclass','Sex','Cabin']).Age.transform(lambda x: x.fillna(x.mean()))

t.Embarked=t.groupby(['Pclass','Sex','Cabin']).Embarked.transform(lambda x: x.fillna(x.mode()))

t['Family Size']=t.SibSp +t.Parch +1

#clean up before ML
t=t.drop(['PassengerId','SibSp','Parch','Ticket'],axis=1)
t=t.dropna()
#print(t.isnull().any())

#noramlize fare to control outliers.
#t.Fare=normalize(t.Fare.values.reshape(len(t.Fare),1))
t.Age=normalize(t.Age.values.reshape(len(t.Age),1))

t=pd.get_dummies(t)
X=t.drop('Survived',axis=1)
y=t.Survived



#ML
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,y_train)
random_forest.predict(X_test)
print("Random Forest: {}".format(random_forest.score(X_test,y_test)))

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
logreg.predict(X_test)
print("logreg: {}".format(logreg.score(X_test,y_test)))




