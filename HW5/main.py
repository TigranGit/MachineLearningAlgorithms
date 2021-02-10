import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ensemble_methods import *
from sklearn import datasets
data = datasets.load_digits(n_class=2) 

X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,
                                                    random_state=0)

# model = Bagging(base_estimator=DecisionTreeClassifier, nr_estimators=20)
# model.fit(X_train, y_train)
# print('Train accuracy: ', accuracy_score(model.predict(X_train), y_train))
# print('Test accuracy: ', accuracy_score(model.predict(X_test), y_test))

estimators = [RandomForestClassifier(), DecisionTreeClassifier()]
# model = WeightedVoting(estimators)
# model.fit(X_train, y_train)
# print('Train accuracy: ', accuracy_score(model.predict(X_train), y_train))
# print('Test accuracy: ', accuracy_score(model.predict(X_test), y_test))

model = Stacking(estimators, 
                Bagging(base_estimator=DecisionTreeClassifier,
                        nr_estimators=20),
                meta_features='prob', cv=False, k=5)
model.fit(X_train, y_train)
print('Train accuracy: ', accuracy_score(model.predict(X_train), y_train))
print('Test accuracy: ', accuracy_score(model.predict(X_test), y_test))
