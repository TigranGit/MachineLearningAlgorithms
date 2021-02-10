import numpy as np
import pandas as pd

class BinaryLDA:
  """
	Binary Linear Discriminant Analysis (LDA) classifier for classification,
	where the labels are encoded as 0s and 1s
  """
  def __init__(self):
    self.w = None
    self.t = None

  def get_covariance_matrix(self, X, N):
    """
    Calculate the covariance matrix for the dataset X
    """
    # It is recommended that you try to compute the covariance matrix
    # by performing the needed operations by hand, instead of using np.cov()
    # function, this way you will have better idea of the covariance matrix

    # Also take into account that the random variables for which you want to
    # compute the covariance matrix are the columns of X, not the rows!

    X_minus_min = X.T - X.mean(axis=0).to_numpy()[:, None]
    covariance_matrix = np.dot(X_minus_min, X_minus_min.T) / float(N - 2)
    return covariance_matrix

  def fit(self, X, y):
      # Separate data by class for convenience
      X1 = X[y == 0]
      X2 = X[y == 1]
      N = X.shape[0]

      # Calculate the covariance matrices of the two datasets
      cov1 = self.get_covariance_matrix(X1, N)
      cov2 = self.get_covariance_matrix(X2, N)

      # cov1 and cov2 should already be normalized,
      # therefore we just add them to get sigma (from our lecture slides)
      sigma = cov1 + cov2

      # Calculate the mean of the two datasets (mu_k in our lecture slides)
      mean1 = X1.mean(axis=0).values
      mean2 = X2.mean(axis=0).values
      mean_diff = mean1 - mean2
      mean_sum = mean1 + mean2

      # Calculate the class priors
      p1 = X1.shape[0] / N
      p2 = X2.shape[0] / N

      # Get the inverse of sigma
      sigma_inv = np.linalg.inv(sigma)

      # determine the decision boundary w*x=t
      # using the formula from the lecture slides
      # you need to figure out which terms form 'w' and
      # which form 't'
      self.w = sigma_inv @ mean_diff
      self.t = 1/2 * mean_sum @ self.w - np.log(p1/p2)

  def predict(self, X):
      # use self.w and self.t to make the predictions for
      # each instance in X
      # (after writing the code with a loop, try to do the same with one line)
      # y_pred = [np.dot(self.w, X.iloc[i]) < self.t for i in range(X.shape[0])]
      y_pred = ((X * self.w).sum(axis=1) < self.t).astype(int)
      return y_pred

class LDA:
  """
	Linear Discriminant Analysis (LDA) classifier
  for multiclass classification with arbitrary label encoding
  """
  def __init__(self):
    self.N = 0
    self.K = 0
    self.classes = None
    self.deltas = None

  def get_covariance_matrix(self, X):
    """Calculate the covariance matrix for the dataset X """
    # It is recommended that you try to compute the covariance matrix
    # by performing the needed operations by hand, instead of using np.cov()
    # function, this way you will have better idea of the covariance matrix

    # Also take into account that the random variables for which you want to
    # compute the covariance matrix are the columns of X, not the rows!

    # don't forget the 1/(N-K) term in this case
    X_minus_min = X.T - X.mean(axis=0)[:, None]
    covariance_matrix = np.dot(X_minus_min, X_minus_min.T) / float(self.N - self.K)
    return covariance_matrix

  def fit(self, X, y):
    # compute means (mu_k), priors (p_k) for each class and
    # sigma by adding the class covariance matrices
    self.classes = np.unique(y)

    self.N = X.shape[0]
    self.K = self.classes.shape[0]

    sigma = sum([self.get_covariance_matrix(X[y == class_]) for class_ in self.classes])

    # get the inverse of sigma
    sigma_inv = np.linalg.inv(sigma)

    # remember that we need to compute the values of the
    # discriminant functions (deltas) for each class, so we will need
    # the respective coefficients to use in the 'predict' method
    X_df = pd.DataFrame(X)
    means = (X_df.groupby(y).mean()).T.values
    priors = (X_df.groupby(y).size() / self.N).T.values

    ws = sigma_inv @ means
    ts_matrix = -1/2 * means.T @ ws + np.log(priors)
    ts = np.diag(ts_matrix)
    # OR
    # for j in range(ws.shape[1]):
    #   w = ws.iloc[:, j].values
    #   mean = means.iloc[:, j].values
    #   t = -1/2 * mean.T @ w + np.log(priors.iloc[j])

    # you can store those coefficients in some data structure
    self.deltas = {'ws': ws, 'ts': ts}

  def predict(self, X):
    # use the coefficients in self.deltas to compute delta_k per class
    # for each instance from X and select the class
    # which has the highest delta_k
    delta_ks = []
    for j in range(self.deltas['ws'].shape[1]):
      w = self.deltas['ws'][:, j]
      t = self.deltas['ts'][j]
      delta_k = (X * w).sum(axis=1) + t
      delta_ks.append(delta_k)
    delta_ks = np.array(delta_ks)
    y_pred = self.classes[delta_ks.argmax(axis=0)]
    return y_pred


class QDA:
  """
	Quadratic Discriminant Analysis (QDA) classifier
  for multiclass classification with arbitrary label encoding
  """
  def __init__(self):
    self.classes = None
    self.deltas = dict()

  def get_covariance_matrix(self, X):
    """ Calculate the covariance matrix for the dataset X """
    # what should be instead of K in this term 1/(N-K) in this case ?!
    X_minus_min = X.T - X.mean(axis=0)[:, None]
    covariance_matrix = np.dot(X_minus_min, X_minus_min.T) / float(len(X) - 1)
    return covariance_matrix

  def fit(self, X, y):
    # compute means (mu_k), priors (p_k) and sigma_k for each class
    # you will also need the determinant and inverse of each sigma_k
    self.classes = np.unique(y)

    sigmas = [self.get_covariance_matrix(X[y == class_]) for class_ in self.classes]
    sigmas_inv = [np.linalg.inv(sigma) for sigma in sigmas]

    # remember that we need to compute the values of the
    # discriminant functions (deltas) for each class, so we will need
    # the respective coefficients to use in the 'predict' method
    X_df = pd.DataFrame(X)
    means = (X_df.groupby(y).mean()).T.values
    priors = (X_df.groupby(y).size() / X.shape[0]).T.values

    # you can store those coefficients in some data structure
    self.deltas['ts'] = [-1/2 * np.log(np.linalg.det(sigma)) + np.log(prior) for sigma, prior in zip(sigmas, priors)]
    self.deltas['means'] = means
    self.deltas['sigmas_inv'] = sigmas_inv

  def predict(self, X):
    # use the coefficients in self.deltas to compute delta_k per class
    # for each instance from X and select the class
    # which has the highest delta_k
    y_pred = []
    for i in range(X.shape[0]):
      x = X[i]
      delta_ks = []
      for j in range(len(self.deltas['ts'])):
        t = self.deltas['ts'][j]
        mean = self.deltas['means'][:, j]
        sigma_inv = self.deltas['sigmas_inv'][j]
        x_minus_mean = x - mean
        delta_k = -1/2 * x_minus_mean.T @ sigma_inv @ x_minus_mean + t
        delta_ks.append(delta_k)
      delta_ks = np.array(delta_ks)
      y_pred.append(delta_ks.argmax(axis=0))
    return self.classes[y_pred]
