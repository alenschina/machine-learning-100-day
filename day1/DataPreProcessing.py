# Step 1: Importing the required libraries

import numpy as np
import pandas as pd

# Step 2: Importing the dataset

# Using the read_csv method of the pandas library to read a local csv file as a dataframe
dataset = pd.read_csv('Data.csv')

# Making separate Matrix and Vector of independent and dependent variables from the dataframe

# Create independent variable（独立变量）. 第一个冒号是所有行（row），第二个是除了最后一个的所有列（column）
X = dataset.iloc[:, :-1].values
# Create dependent variable（依赖变量）. 所有行的第三列
y = dataset.iloc[:, 3].values

# Handle the missing data
from sklearn.preprocessing import Imputer

# 为了避免数据缺失对机器学习模型构造造成影响，我们需要处理数据中的缺失项。通常情况下使用数据的平均值或者中间值填充缺失项，这里我们使用平均值
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encode categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Splotting the dataset into test set and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train)
print(X_test)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print(X_train)
print(X_test)