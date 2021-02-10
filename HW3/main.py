import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from svm import SVM

def synthetic_dataset():
    '''Generate 3 Gaussian samples with different covariance matrices'''
    n, dim = 300, 2
    np.random.seed(0)
    cov1 = np.array([[1, -1],
                     [-1, 2]])
     
    cov2 = np.array([[0.5, 0.2],
                     [0.2, 0.5]])
    
    cov3 = np.array([[2, -0.5],
                     [-0.5, 0.2]])

    # inputs
    X = np.vstack((np.random.multivariate_normal(mean=[0, 0], cov=cov1, size=n),
                   np.random.multivariate_normal(mean=[2, 2], cov=cov2, size=n),
                   np.random.multivariate_normal(mean=[-3, 3], cov=cov3, size=n)
                   ))
    # labels 0, 1, 2
    y = np.hstack((np.zeros(n), np.ones(n), np.ones(n) * 2))

    return X, y

Xs, ys = synthetic_dataset()

X_train, X_test, y_train, y_test = train_test_split(Xs[ys!=0], ys[ys!=0],
                                                    test_size=0.25,
                                                    random_state=0)
y_train[y_train == 2] = -1
y_test[y_test == 2] = -1

model = SVM(C=None)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy_score(predictions, y_test))

# model = SVM(C=0.01)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# print(accuracy_score(predictions, y_test))

model = SVM(C=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy_score(predictions, y_test))

# x1 = np.array([5, 6, 7, 7, 8, 9, 0, 1, 2, 4, 5, 6])
# x2 = np.array([2, 1, 3, 5, 10, 3, 4, 8, 6, 10, 9, 11])
# y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
# data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# X = data.iloc[:,:-1].values
# y = data.iloc[:, -1].values.copy()
# y[y==0] = -1  # our implementation requires 1, -1 encoding of labels

# model = SVM(C=0)
# model.fit(X, y)
