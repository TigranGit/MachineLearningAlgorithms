import numpy as np
from sklearn.model_selection import train_test_split

from gradient_descents import *


x = np.random.random(size=10000)
y = np.sqrt(1 + x)

ones = np.ones(x.shape[0])[:,np.newaxis]
X = np.concatenate([ones, x[:,np.newaxis]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


w = gradient_descent(X_train, y_train, learning_rate=0.0001)

print('Gradient Descent MSE:', mse(y_test, X_test @ w))
print()

w = stochastic_gradient_descent(X_train, y_train, learning_rate=0.001)
print('Stochastic Gradient Descent MSE:', mse(y_test, X_test @ w))
print()

w = mini_batch_gradient_descent(X_train, y_train, learning_rate=1e-3, batch_size=10, max_epoch=15)
print('Mini-Batch Gradient Descent MSE:', mse(y_test, X_test @ w))
print()
