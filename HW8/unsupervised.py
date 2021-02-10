import numpy as np
from numpy.core.defchararray import center
from numpy.core.numeric import indices

class KMeans:
  def __init__(self, k=2, max_iterations=500, tol=0.5):
    # number of clusters
    self.k = k
    # maximum number of iterations to perform
    # for updating the centroids
    self.max_iterations = max_iterations
    # tolerance level for centroid change after each iteration
    self.tol = tol
    # we will store the computed centroids
    self.centroids = None

  def init_centroids(self, X):
    # this function initializes the centroids
    # by choosing self.k points from the dataset
    # Hint: you may want to use the np.random.choice function
    centroids = np.random.choice(np.arange(X.shape[0]), size=self.k)
    return X[centroids]

  def closest_centroid(self, X):
    # this function computes the distance (euclidean) between
    # each point in the dataset from the centroids filling the values
    # in a distance matrix (dist_matrix) of size n x k
    # Hint: you may want to remember how we solved the warm-up exercise
    # in Programming module (Python_Numpy2 file)
    dist_matrix = np.array([np.sqrt(((X - centroid) ** 2).sum(1)) for centroid in self.centroids]).T
    # after constructing the distance matrix, you should return
    # the index of minimal value per row
    # Hint: you may want to use np.argmin function
    return np.argmin(dist_matrix, axis=1)

  def update_centroids(self, X, label_ids):
    # this function updates the centroids (there are k centroids)
    # by taking the average over the values of X for each label (cluster)
    # here label_ids are the indices returned by closest_centroid function

    new_centroids = np.empty(self.centroids.shape)
    for i, label_id in enumerate(np.unique(label_ids)):
      X_label = X[label_ids == label_id]
      new_centroid = X_label.mean(axis=1)
      new_centroids[i] = new_centroid

    return new_centroids

  def fit(self, X):
    # this is the main method of this class
    X = np.array(X)
    # we start by random centroids from our data
    self.centroids = self.init_centroids(X)

    not_converged = True
    i = 1 # keeping track of the iterations
    while not_converged and (i < self.max_iterations):
      current_labels = self.closest_centroid(X)
      new_centroids = self.update_centroids(X, current_labels)

      # count the norm between new_centroids and self.centroids
      # to measure the amount of change between
      # old cetroids and updated centroids
      norm = np.linalg.norm(self.centroids - new_centroids)
      not_converged = norm > self.tol
      self.centroids = new_centroids
      i += 1
    self.labels = current_labels
    print(f'Converged in {i} steps')

  def predict(self, X):
    # we can also have a method, which takes a new instance (instances)
    # and assigns a cluster to it, by calculating the distance
    # between that instance and the fitted centroids
    # returns the index (indices) of the cluster labels for each instance
    X = np.array(X)
    return self.closest_centroid(X)

class HierarchicalClustering:
  def __init__(self, nr_clusters, diss_func, linkage='single', distance_threshold=None):
    # nr_clusters is the number of clusters to find from the data
    # if distance_treshold is None, nr_clusters should be provided
    # and if distance_threshold is provided, then we stop
    # forming clusters when we reach the specified threshold
    # diss_func is the dissimilarity measure to compute the
    # dissimilarity/distance between two data points
    # linkage method should be one of the following {single, complete, average}
    self.nr_clusters = nr_clusters
    self.diss_func = diss_func
    self.distance_threshold = distance_threshold
    self.linkage = linkage

    self.clusters = []

  def get_distances(self, m, n=None):
    set_nan = False
    if n is None:
      n = m
      set_nan = True

    diss_matrix = np.empty((len(m), len(n)))
    for i in range(len(m)):
      diss_matrix[i] = self.diss_func(m[i], n)
      if set_nan and i < len(n):
        diss_matrix[i, i] = np.nan
    return diss_matrix

  def fit(self, X_org):
    if isinstance(X_org, np.ndarray):
      X = X_org.copy()
    else:
      X = X_org.to_numpy()

    while 1:
      # Point to point
      if len(X) >= 2:
        diss_matrix = self.get_distances(X)
        i, j = np.unravel_index(np.nanargmin(diss_matrix), diss_matrix.shape)
        distance = diss_matrix[i, j]
      else:
        distance = np.inf

      # Point to cluster
      if len(X) >= 1:
        near_cluster_index = -1
        for cluster_i, cluster in enumerate(self.clusters):
          if self.linkage == 'average':
            cluster_center = np.mean(cluster, axis=0)
            diss_matrix = self.get_distances(X, cluster_center.reshape(1, -1))
            k = np.nanargmin(diss_matrix)
            cluster_distance = diss_matrix[k]
          else:
            diss_matrix = self.get_distances(X, cluster)
            if self.linkage == 'single':
              k, p = np.unravel_index(np.nanargmin(diss_matrix), diss_matrix.shape)
              cluster_distance = diss_matrix[k][p]
            elif self.linkage == 'complete':
              cluster_distance = np.min(np.max(diss_matrix, axis=1))
              k = np.where(diss_matrix == cluster_distance)[0]
            else:
              raise ValueError('Invalid linkage')

          if cluster_distance < distance:
            distance = cluster_distance
            i = k
            near_cluster_index = cluster_i

      # Cluster to cluster
      near_cluster_i = near_cluster_j = -1
      cluster_distance = distance + 1
      if self.clusters and len(self.clusters) >= 2:
        # diss_matrix = np.empty([len(self.clusters), len(self.clusters)])
        for cluster_i in range(len(self.clusters)):
          for cluster_j in range(len(self.clusters)):
            if cluster_i == cluster_j:
              continue
            if self.linkage == 'average':
              cluster_center_i = np.mean(self.clusters[cluster_i], axis=0).reshape(1, -1)
              cluster_center_j = np.mean(self.clusters[cluster_j], axis=0).reshape(1, -1)
              new_cluster_distance = self.diss_func(cluster_center_i, cluster_center_j)
            else:
              diss_matrix = self.get_distances(self.clusters[cluster_i], self.clusters[cluster_j])
              if self.linkage == 'single':
                new_cluster_distance = np.nanmin(diss_matrix)
              elif self.linkage == 'complete':
                new_cluster_distance = np.max(diss_matrix)
              else:
                raise ValueError('Invalid linkage')
            if new_cluster_distance < cluster_distance:
              cluster_distance = new_cluster_distance
              near_cluster_i = cluster_i
              near_cluster_j = cluster_j

      if self.distance_threshold:
        if distance >= self.distance_threshold and cluster_distance >= self.distance_threshold:
          break

      if cluster_distance < distance:
        # Cluster to cluster
        cluster = np.append(self.clusters[near_cluster_i], self.clusters[near_cluster_j], axis=0)
        if near_cluster_i > near_cluster_j:
          del self.clusters[near_cluster_i]
          del self.clusters[near_cluster_j]
        else:
          del self.clusters[near_cluster_j]
          del self.clusters[near_cluster_i]
        self.clusters.append(cluster)
      elif near_cluster_index != -1:
        # Point to cluster
        self.clusters[near_cluster_index] = np.append(self.clusters[near_cluster_index], X[i].reshape(1, -1), axis=0)
        X = np.delete(X, i, axis=0)
      else:
        # Point to point
        cluster = np.array([X[i], X[j]])
        if i > j:
          X = np.delete(X, i, axis=0)
          X = np.delete(X, j, axis=0)
        else:
          X = np.delete(X, j, axis=0)
          X = np.delete(X, i, axis=0)
        self.clusters.append(cluster)

      # print('self.clusters')
      # print(self.clusters)

      if len(X) == 0 and (len(self.clusters) == self.nr_clusters or len(self.clusters) == 1):
        break

  def predict(self, X):
    if isinstance(X, np.ndarray):
      X = X.copy()
    else:
      X = X.to_numpy()

    y = np.empty(X.shape[0])
    for i, point in enumerate(X):
      distance = np.inf
      near_cluster_index = None
      for cluster_i, cluster in enumerate(self.clusters):
        if self.linkage == 'average':
          cluster_center = np.mean(cluster, axis=0)
          cluster_distance = self.diss_func(point, cluster_center.reshape(1, -1))
        else:
          diss_matrix = self.get_distances(point, cluster)
          if self.linkage == 'single':
            k, p = np.unravel_index(np.nanargmin(diss_matrix), diss_matrix.shape)
            cluster_distance = diss_matrix[k][p]
          elif self.linkage == 'complete':
            cluster_distance = np.min(np.max(diss_matrix, axis=1))
          else:
            raise ValueError('Invalid linkage')

        if cluster_distance < distance:
          distance = cluster_distance
          near_cluster_index = cluster_i
      y[i] = near_cluster_index
    return y

class DBSCAN:
  def __init__(self, diss_func, epsilon=0.5, min_points=5):
    # epsilon is the maximum distance/dissimilarity between two points
    # to be considered as in the neighborhood of each other
    # min_ponits is the number of points in a neighborhood for
    # a point to be considered as a core point (a member of a cluster).
    # This includes the point itself.
    # diss_func is the dissimilarity measure to compute the
    # dissimilarity/distance between two data points
    self.diss_func = diss_func
    self.epsilon = epsilon
    self.min_points = min_points

    self.clusters = []
    self.noise = []

  def flatten_array(self, array):
    if isinstance(array, list):
        return [a for i in array for a in self.flatten_array(i)]
    else:
        return [array]

  def get_distances(self, m, n=None):
    set_nan = False
    if n is None:
      n = m
      set_nan = True

    diss_matrix = np.empty((len(m), len(n)))
    for i in range(len(m)):
      diss_matrix[i] = self.diss_func(m[i], n)
      if set_nan and i < len(n):
        diss_matrix[i, i] = np.nan
    return diss_matrix

  def get_near_points(self, point, X):
    diss_array = self.diss_func(point, X)
    near_points = X[diss_array < self.epsilon]
    return near_points

  def find_near_points(self, point, X):
    near_points = self.get_near_points(point, X)
    if len(near_points) < 2:
      return point
    X_minus_point = np.delete(X, np.where(X == point), axis=0)
    all_near_points = [point]
    for near_point in near_points:
      if (near_point == point).all():
        continue
      all_near_points.append(self.find_near_points(near_point, X_minus_point))
    return all_near_points

  def fit(self, X_org):
    # noise should be labeled as "-1" cluster
    if isinstance(X_org, np.ndarray):
      X = X_org.copy()
    else:
      X = X_org.to_numpy()

    # Form initial clusters
    while len(X) > 0:
      point = X[0]
      near_points = self.get_near_points(point, X)
      if len(near_points) >= self.min_points:
        all_near_points_list = self.find_near_points(point, X)
        all_near_points = np.unique(np.array(self.flatten_array(all_near_points_list)), axis=0)
        self.clusters.append(all_near_points)
        index_list = []
        for near_point in all_near_points:
          index_list.append(np.where((X == near_point).all(axis=1))[0][0])
        X = np.delete(X, index_list, axis=0)
      else:
        self.noise.append(point)
        X = np.delete(X, 0, axis=0)

    # Check if there is a noise near some cluster point
    if self.noise:
      for cluster in self.clusters:
        point = cluster[0]
        all_near_points_list = self.find_near_points(point, np.append(cluster, np.array(self.noise), axis=0))
        all_near_points = np.unique(np.array(self.flatten_array(all_near_points_list)), axis=1)
        for near_point in all_near_points:
          if not (near_point == cluster).all(axis=1).all():
            cluster = np.append(cluster, [near_point], axis=0)
            for i, noise in enumerate(self.noise):
              if (near_point == noise).all():
                del self.noise[i]

  def predict(self, X_org):
    if isinstance(X_org, np.ndarray):
      X = X_org.copy()
    else:
      X = X_org.to_numpy()
    y = np.full(len(X), -1)
    for i, cluster in enumerate(self.clusters):
      cluster_point = cluster[0]
      all_near_points_list = self.find_near_points(cluster_point, np.append(cluster, np.array(self.noise), axis=0))
      all_near_points = np.unique(np.array(self.flatten_array(all_near_points_list)), axis=1)
      for near_point in all_near_points:
        if not (near_point == cluster).all(axis=1).all():
          for j, point in enumerate(X):
            if (near_point == point).all():
              y[j] = i
    return y


class PCA:
  def __init__(self, nr_components):
    self.nr_components = nr_components

    # we will store the PC coordinates here
    self.components = None
    # how much variance is explained with the PCs
    self.explained_variance = None
    # how much variance is explained with the PCs among the total variance
    self.explained_variance_ratio = None

  def fit(self, X):
    # this method is used to compute the PC components (projection matrix)
    nr_components = self.nr_components

    # compute the covariance matrix of the given dataset
    # note that we are interested in covariance in terms of the
    # features (columns) of our dataset
    Z = X - X.mean(axis=0)
    covariance_matrix = (Z.T @ Z) / nr_components

    # get the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # get the indices of the first nr_components eigenvalues
    idx = np.arange(nr_components)
    self.explained_variance = sum(eigenvalues[idx][:nr_components])
    self.explained_variance_ratio = self.explained_variance / sum(eigenvalues)

    # select the first nr_components eigenvectors as the projection matrix
    self.components = eigenvectors[:nr_components]

  def transform(self, X):
    # this method will project the initial data to the new subspace
    # spanned with the principal components, here you will need self.components
    return X @ self.components.T

  def fit_transform(self, X):
    self.fit(X)
    return self.transform(X)
