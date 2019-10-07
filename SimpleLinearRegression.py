# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")

#Divide dataset into x and y
#first : - all the rows
#second : - all the rows -1 which means dropping salary column
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

#splitting data based on training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
#implement classifier based on simple linear regression
from sklearn.linear_model import LinearRegression
simplelinearRegression=LinearRegression()
simplelinearRegression.fit(X_train,Y_train)

y_predict=simplelinearRegression.predict(X_test)

y_predict_val=simplelinearRegression.predict(11)

#implement the graph

plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,simplelinearRegression.predict(X_train))
plt.show()