from numpy.lib.polynomial import poly
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from svm_with_kernels import SupportVectorMachine

df = pd.read_csv('train.csv')
data = df[['Pclass', 'Sex', 'Survived']]
data_with_age = df[['Pclass', 'Sex', 'Age', 'Survived']]
data_with_age = data_with_age.dropna()
X_train, X_test, y_train, y_test = train_test_split(data_with_age.drop(columns='Survived'),
                                                    data_with_age.Survived,
                                                    random_state=0)

data1 = pd.get_dummies(data, drop_first=True)
X_train1, X_test1, y_train, y_test = train_test_split(data1.drop(columns='Survived'),
                                                      data1.Survived,
                                                      random_state=0)
data2 = pd.get_dummies(data_with_age, drop_first=True)
X_train2, X_test2, y_train, y_test = train_test_split(data2.drop(columns='Survived'),
                                                      data2.Survived,
                                                      random_state=0)

y_train[y_train==0] = -1  # our implementation requires 1, -1 encoding of labels
y_test[y_test==0] = -1

# def polynomial(x1, x2): return (np.dot(0.333333 * x1, x2.T) + 2) ** 2
def polynomial(x1, x2, p=2):
    return (np.dot(2 * x1, x2.T) + 10) ** p
def linear(x1, x2): return np.dot(x1, x2.T)
def rbf(x1, x2, gamma=10): return np.exp(-gamma * (np.linalg.norm(x1 - x2) ** 2))
def gaussian(x, z, gamma=10):
    return np.exp((np.linalg.norm(x - z, axis=1) ** 2) * -gamma)

print('MY:')
model = SupportVectorMachine(kernel_name='linear')
model.fit(X_train2, y_train)
prediction = model.predict(X_test2)
print(accuracy_score(prediction, y_test))

model = SupportVectorMachine(kernel_name='poly', power=1, coef=0, gamma=1)
model.fit(X_train2, y_train)
prediction = model.predict(X_test2)
print(accuracy_score(prediction, y_test))
print('-----------------')
"""
self.t = 0
for i in range(len(self.alphas)):
  self.t += self.alphas[i] * self.support_vector_labels[i] * kernel_matrix[ind[i], 0]
self.t -= self.support_vector_labels[0]
"""
# OR
"""
self.t = 0
for n in range(len(self.alphas)):
  self.t += np.sum(self.alphas * self.support_vector_labels * kernel_matrix[ind[n],idx])
  self.t -= self.support_vector_labels[n]
self.t /= len(self.alphas)
"""

from utils import create_dataset, plot_contour

np.random.seed(1)
X, y = create_dataset(N=50)

from sklearn.svm import SVC

model = SupportVectorMachine(kernel_name='rbf', power=2, coef=2, gamma=1)
model.fit(X, y)
y_pred = model.predict(X)
print('Acc:', accuracy_score(np.array(y_pred), np.array(y)))
plot_contour(X, y, model)


model = SVC(kernel='rbf', degree=2, coef0=2, gamma=1)
model.fit(X, y)
y_pred = model.predict(X)
print('Acc:', accuracy_score(np.array(y_pred), np.array(y)))
plot_contour(X, y, model)
