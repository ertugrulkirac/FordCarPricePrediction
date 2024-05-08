# -*- coding: utf-8 -*-
"""
Created on Wed May  8 01:40:36 2024

@author: ertugrulkirac
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression

data = pd.read_csv('ford.csv')

print(data.head())

print(data.tail(15))

print(data.info())

print(data.shape)

#categorical_columns = ['model','transmission','fuelType']

dfdum = pd.get_dummies(data,columns=['model','transmission','fuelType'],drop_first=True)

y=dfdum[['price']]
x=dfdum.drop("price",axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.70,random_state=0)

reg=LinearRegression()
model=reg.fit(x_train,y_train)
y_pred = model.predict(x_test)
score = model.score(x_test,y_test)
print(score*100)