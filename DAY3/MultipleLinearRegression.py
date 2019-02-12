# Step 1: Data Preprocessing

# import the libraries
import pandas as pd
import numpy as np

# import the dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values


# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])
print(X)
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()


# avoiding dummy variable trap
X = X[:, 1:]

# splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

# Step 2: Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

print(y_pred)



