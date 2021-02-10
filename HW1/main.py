import pandas as pd
from knn import KNearestNeighbor
from naive_bayes import MyNaiveBayes
from sklearn.model_selection import train_test_split

def accuracy(actual, predicted):
    return sum(actual == predicted) / len(predicted)

data = pd.read_csv("car.csv", dtype="category", header = None)
data.columns = ["buying", "maint", "doors", "persons",
                "lug-boot", "safety", "accept"]

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1],
                                                    data['accept'],
                                                    test_size=0.25,
                                                    random_state=0)

from knn import KNearestNeighbor
model = KNearestNeighbor(X_train, y_train)
predictions = model.fit_predict(X_test, k=1)
print(accuracy(y_test, predictions))

# model = MyNaiveBayes(smoothing=False)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# print(accuracy(y_test, predictions))

# model = MyNaiveBayes(smoothing=True)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# print(accuracy(y_test, predictions))
