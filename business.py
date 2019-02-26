# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:24:19 2019

@author: k98p
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Artificial_Neural_Networks/Churn_Modelling.csv')
X=dataset.iloc[:, 3:13].values
Y=dataset.iloc[:, 13].values
print(X)

#encoding categorical data, such as giving numbers to strings.
#since there are 3 countires in the country column, and 2 choices in the gender, we'll use encoding for these
#don't use encoding for names, IDs as they can be and should be unique and may give rise to huge figures after encoding

from sklearn.prepocessing import LabelEncoder as LE
from sklearn.prepocessing import OneHotEncoder as OHE
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fot_transform(X).toarray()
X=X[:,1:]

#Feature Scaling
#getting the training set and test set
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit.transform(X_train)
X_test = sc.transform(X_test)

#importing keras libraries and packages
import keras
form keras.models import Sequential
form keras.layers import Dense

#initializing the ANN
classifier = Sequential()


