import pandas as pd
import numpy as np
# import sklearn.datasets

from pprint import pprint
from unsupervised import KMeans, HierarchicalClustering, DBSCAN

data = pd.DataFrame(
    {'ID':['P1','P2','P3','P4','P5','P6','P7','P8','P9'],
    'x':[0,1,4,2,0,1,3,7,6],
    'y':[1,1,0,3,5,6,6,4,8]})
X = np.array(data[['x', 'y']])

# model = KMeans(k=2, tol=0.001)
# model.centroids = model.init_centroids(X)
# current_labels = model.closest_centroid(X)
# print(current_labels)


# diss_func = lambda a, b: np.sqrt(np.sum(np.power(a - b, 2), axis=-1))
diss_func = lambda a, b: np.linalg.norm(a - b, axis=1)
# model = HierarchicalClustering(3, diss_func, linkage='average')
# model.fit(X)

# print(model.predict(X))

model = DBSCAN(diss_func, epsilon=3, min_points=2)
model.fit(X)
pprint(model.clusters)
pprint(model.noise)

pprint(model.predict(X))