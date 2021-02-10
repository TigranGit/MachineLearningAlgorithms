import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso

data = pd.read_csv("data.csv", index_col = 0)
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['y']),
                                                    data['y'], test_size=0.2,
                                                    random_state=0 )

from regression import MyLinearRegression

model = MyLinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
our_mse = mean_squared_error(preds, y_test)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
sklearn_mse = mean_squared_error(pred, y_test)
print(our_mse, '-' ,sklearn_mse)
assert np.round(our_mse, 2) == np.round(sklearn_mse, 2)

# Ridge
model = MyLinearRegression(regularization='l2', lam=1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('Our mse:', mean_squared_error(preds, y_test))

model = Ridge()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('Sklearn mse:', mean_squared_error(pred, y_test))

# Lasso
model = MyLinearRegression(regularization='l1',
                         lam=1, 
                         learning_rate=0.001,
                         tol=0.005)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('Our mse:', mean_squared_error(preds, y_test))



# from regression import SVR
# model = SVR()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print('SVR mse:', mean_squared_error(pred, y_test))

# from regression import RegressionTree
# model = RegressionTree()
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print('Regression tree mse:', mean_squared_error(pred, y_test))
