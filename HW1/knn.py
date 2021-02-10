import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with euclidean distance """

  def __init__(self, X_train, y_train):
    """
    Initializing the KNN object

    Inputs:
    - X_train: A numpy array or pandas DataFrame of shape (num_train, D)
    - y_train: A numpy array or pandas Series of shape (N,) containing the training labels
    """
    self.X_train = X_train
    self.y_train = y_train

  def fit_predict(self, X_test, k=1):
    """
    This method fits the data and predicts the labels for the given test data.
    For k-nearest neighbors fitting (training) is just
    memorizing the training data.
    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D)
    - k: The number of nearest neighbors.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing predicted labels
    """
    dists = self.compute_distances(X_test)
    return self.predict_labels(dists, k=k)

  def compute_distances(self, X_test):
    """
    Compute the euclidean distance between each test point in X_test and each training point
    in self.X_train.

    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the euclidean distance between the ith test point and the jth training
      point.
    """
    x_1 = self.one_hot_encoding(X_test)
    x_2 = self.one_hot_encoding(self.X_train)

    dists = np.sqrt(np.power(x_1[:,np.newaxis]-x_2, 2).sum(axis=2))
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance between the ith test point and the jth training point.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing the
    predicted labels for the test data
    """
    y_pred = []
    min_dists = dists == dists.min(axis=0)
    for i in range(min_dists.shape[0]):
      ys = self.y_train[min_dists[i]][:k]
      values, counts = np.unique(ys, return_counts=True)
      index = np.argmax(counts)
      y_pred.append(values[index])
    return np.array(y_pred)

  def one_hot_encoding(self, df):
    if isinstance(df, np.ndarray):
      X = df.copy()
    else:
      X = df.to_numpy()

    # Get unique classes.
    classes = np.unique(X)

    # Replace classes with itegers.
    X = np.searchsorted(classes, X)

    # Get an identity matrix.
    eye = np.eye(classes.shape[0])

    # Iterate over all columns
    # and get one-hot encoding for each column.
    X = np.concatenate([eye[i] for i in X.T], axis=len(df.shape)-1)
    return X
