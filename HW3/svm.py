import numpy as np
import cvxopt  # library for convex optimization

# hide cvxopt output
cvxopt.solvers.options['show_progress'] = False

class SVM:
  """
  Hard (C=None) and Soft (C>0) Margin Support Vector Machine classifier (binary)
  """
  def __init__(self, C=1):
    self.C = C
    self.alphas = None
    self.support_vectors = None
    self.support_vector_labels = None
    self.w = None
    self.t = None

  def fit(self, X, y):
    """
    X - numpy array or pandas DataFrame
    y - numpy array or pandas DataFrame with 1 and -1 encoding
    """
    m, n = np.shape(X)
    y = y[:, np.newaxis] * 1.
    X_dash = y * X

    # define the quadratic optimization problem
    # (the dual problem from the lecture slides but in terms of minimization)
    # by constructing the appropriate matrices P, q, A, b...
    P = cvxopt.matrix(np.dot(X_dash , X_dash.T) * 1.)
    q = cvxopt.matrix(-np.ones((m, 1)))
    A = cvxopt.matrix(y.T)
    b = cvxopt.matrix(np.zeros(1))

    if not self.C:
      # the case when C=0 (Hard-margin SVM)
      G = cvxopt.matrix(-np.eye(m))
      h = cvxopt.matrix(np.zeros(m))
    else:
      # the case when C>0 (Soft-margin SVM)
      G_max = np.eye(m) * -1
      G_min = np.eye(m)
      G = cvxopt.matrix(np.vstack((G_max, G_min)))
      h_max = cvxopt.matrix(np.zeros(m))
      h_min = cvxopt.matrix(np.ones(m) * self.C)
      h = cvxopt.matrix(np.vstack((h_max, h_min)))

    # solve the quadratic optimization problem using cvxopt
    minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

    # lagrange multipliers (denoted by alphas in the lecture slides)
    alphas = np.ravel(minimization['x'])

    # first get indexes of non-zero lagr. multipiers
    idx = alphas > 1e-7

    # get the corresponding lagr. multipliers (non-zero alphas)
    self.alphas = alphas[idx]

    # get the support vectors
    self.support_vectors = X[idx]

    # get the corresponding labels
    self.support_vector_labels = y[idx]

    # calculate w using the alphas, support_vectors and
    # the corresponding labels
    self.w = ((self.alphas[:, np.newaxis] * self.support_vector_labels).T @ self.support_vectors)[0]

    # calculate t using w and the first support vector
    self.t = np.dot(self.w, self.support_vectors[0]) - self.support_vector_labels[0]

  def predict(self, X):
    return np.sign(self.w @ X.T - self.t)