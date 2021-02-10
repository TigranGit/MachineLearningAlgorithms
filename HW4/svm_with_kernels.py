import numpy as np
import cvxopt  # library for convex optimization
# hide cvxopt output
cvxopt.solvers.options['show_progress'] = False

class SupportVectorMachine:
  """
  Hard (C=0) and Soft (C>0) Margin Support Vector Machine classifier
  with kernels
  """
  def __init__(self, C=1, kernel_name='linear', power=2, gamma=None, coef=2):
    self.C = C
    self.power = power # degree of the polynomial kernel (d in the slides)
    self.gamma = gamma # Kernel coefficient for "rbf" and "poly"
    self.coef = coef # coefficent of the polynomial kernel (r in the slides)
    self.kernel_name = kernel_name  # implement for 'linear', 'poly' and 'rbf'
    self.kernel = None
    self.alphas = None
    self.support_vectors = None
    self.support_vector_labels = None
    self.t = None

  def get_kernel(self, kernel_name):
    # you can define the three kernel functions under this method
    # and then use a dictionary with keys being the names of the kernels
    # and the values being the kernel functions with respective parameters

    def linear(x1, x2): return np.dot(x1, x2.T)
    def polynomial(x1, x2): return (np.dot(self.gamma * x1, x2.T) + self.coef) ** self.power
    def rbf(x1, x2): return np.exp(-self.gamma * (np.linalg.norm(x1 - x2) ** 2))

    kernel_functions = {'linear': linear,
                        'poly': polynomial,
                        'rbf': rbf}

    return kernel_functions[kernel_name]

  def fit(self, X, y):
    if not isinstance(X, np.ndarray):
      X = np.array(X, dtype='float')
    if not isinstance(y, np.ndarray):
      y = np.array(y, dtype='float')

    nr_samples, nr_features = np.shape(X)

    # Setting a default value for gamma
    if not self.gamma:
      self.gamma = 1 / nr_features

    # Set the kernel function
    self.kernel = self.get_kernel(self.kernel_name)

    # Construct the kernel matrix
    kernel_matrix = np.zeros((nr_samples, nr_samples))
    for i in range(nr_samples):
      for j in range(nr_samples):
        kernel_matrix[i, j] = self.kernel(X[i], X[j])

    # Define the quadratic optimization problem
    P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
    q = cvxopt.matrix(np.ones(nr_samples) * -1)
    A = cvxopt.matrix(y, (1, nr_samples), tc='d')
    b = cvxopt.matrix(0, tc='d')

    if not self.C:
      G = cvxopt.matrix(np.identity(nr_samples) * -1)
      h = cvxopt.matrix(np.zeros(nr_samples))
    else:
      G_max = np.identity(nr_samples) * -1
      G_min = np.identity(nr_samples)
      G = cvxopt.matrix(np.vstack((G_max, G_min)))
      h_max = cvxopt.matrix(np.zeros(nr_samples))
      h_min = cvxopt.matrix(np.ones(nr_samples) * self.C)
      h = cvxopt.matrix(np.vstack((h_max, h_min)))

    # Solve the quadratic optimization problem using cvxopt
    minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers (denoted by alphas in the lecture slides)
    alphas = np.ravel(minimization['x'])

    # first get indexes of non-zero lagr. multipiers
    idx = alphas > 1e-7
    ind = np.arange(len(alphas))[idx]

    # get the corresponding lagr. multipliers (non-zero alphas)
    self.alphas = alphas[idx]

    # get the support vectors
    self.support_vectors = X[idx]

    # get the corresponding labels
    self.support_vector_labels = y[idx]

    # Calculate intercept (t) with first support vector
    self.t = 0
    for n in range(len(self.alphas)):
      self.t += np.sum(self.alphas * self.support_vector_labels * kernel_matrix[ind[n],idx])
      self.t -= self.support_vector_labels[n]
    self.t /= len(self.alphas)

  def predict(self, X):
    y_pred = []
    for instance in np.array(X):
      prediction = 0
      # determine the label of the given instance by the support vectors
      for i in range(len(self.alphas)):
        prediction += self.alphas[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], instance)
      prediction -= self.t
      y_pred.append(np.sign(prediction))
    return np.array(y_pred)
