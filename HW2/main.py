import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid') # Plot style

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from discriminants import BinaryLDA, LDA, QDA

def plot_data(data):
  plt.figure(figsize=(8, 6))
  plt.plot(data.loc[data.y==1, 'x1'], data.loc[data.y==1, 'x2'],
          'ro' , label='positives')
  plt.plot(data.loc[data.y==0, 'x1'], data.loc[data.y==0, 'x2'],
          'bo' , label='negative')

  min_x1 = np.min(data['x1'])
  max_x1 = np.max(data['x1'])
  min_x2 = np.min(data['x2'])
  max_x2 = np.max(data['x2'])

  plt.legend(markerscale=1, loc='upper left', 
            frameon=True, shadow=True, fontsize=12)

  plt.xticks(list(range(min_x1-1, max_x2+2)))
  plt.yticks(list(range(min_x2-1, max_x2+2)))
  plt.xlim(min_x1-1, max_x1+1)
  plt.ylim(min_x1-1, max_x2+1)

  plt.xlabel('$x_1$', size=12)
  plt.ylabel('$x_2$', size=12)

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

def plot_data2(X, y, y_pred):
  plt.figure(figsize=(8, 6))

  tp = (y == y_pred)  # True Positive (predicted correctly)
  tp0, tp1, tp2 = tp[y == 0], tp[y == 1], tp[y == 2]
  X0, X1, X2 = X[y == 0], X[y == 1], X[y == 2]
  X0_tp, X0_fp = X0[tp0], X0[~tp0]
  X1_tp, X1_fp = X1[tp1], X1[~tp1]
  X2_tp, X2_fp = X2[tp2], X2[~tp2]

  # class 0: 
  plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
  plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
              s=30, color='#990000')  # dark red

  # class 1: 
  plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
  plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
              s=30, color='#000099')  # dark blue
  
  # class 3: 
  plt.scatter(X2_tp[:, 0], X2_tp[:, 1], marker='.', color='green')
  plt.scatter(X2_fp[:, 0], X2_fp[:, 1], marker='x',
              s=30, color='#009900')  # dark green

# x1 = np.array([6, 6, 7, 7, 8, 9, 0, 1, 2, 4, 5, 5])
# x2 = np.array([11, 1, 3, 5, 10, 3, 4, 8, 6, 10, 9, 2])
# y = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1])

# data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
# X = data.iloc[:,:-1]
# y = data.iloc[:, -1]

# model = BinaryLDA()
# model.fit(X, y)

# print(list(model.predict(X)))
# # print(y)


# clf = LinearDiscriminantAnalysis()
# clf.fit(X, y)
# print(clf.predict(X))

Xs, ys = synthetic_dataset()
X_train, X_test, y_train, y_test = train_test_split(Xs, ys,
                                                    test_size=0.25,
                                                    random_state=0)

model = LDA()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy_score(predictions, y_test))

model = QuadraticDiscriminantAnalysis()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy_score(predictions, y_test))