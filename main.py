#%%codecell
#import dependencies
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

#%%codecell
#import boston data
raw_data = load_boston()
print(raw_data)
X = pd.DataFrame(raw_data['data'], columns = raw_data['feature_names'])
print(X.head())
y = pd.DataFrame(raw_data['target'])
print(y.head())

#%%codecell
#preprocessing
scalar = StandardScaler()
X = pd.DataFrame(scalar.fit_transform(X), columns = raw_data['feature_names'])
print(X.head())

#%%codecell
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#%%codecell
#Training model
reg = LinearRegression()
reg.fit(X_train, y_train)

#%%codecell
#checking accuracy
y_pred = reg.predict(X_test)
print('Mean squared error:', mean_squared_error(y_test, y_pred))
print("accuracy:", reg.score(X_test, y_test))
