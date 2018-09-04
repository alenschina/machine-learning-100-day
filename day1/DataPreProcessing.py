# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Create independent variable（独立变量）. 第一个冒号是所有行（row），第二个是除了最后一个的所有列（column）
X = dataset.iloc[:, :-1].values
# Create dependent variable（依赖变量）. 所有行的第三列
y = dataset.iloc[:, 3].values

# Handle missing data
from sklearn.preprocessing import Imputer

# 为了避免数据缺失对机器学习模型构造造成影响，我们需要处理数据中的缺失项。通常情况下使用数据的平均值或者中间值填充缺失项，这里我们使用平均值
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
