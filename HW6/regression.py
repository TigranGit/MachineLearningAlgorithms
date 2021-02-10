import numpy as np
import cvxpy as cp
import progressbar

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

class MyLinearRegression:

  def __init__(self, regularization=None, lam=0, learning_rate=1e-3, tol=0.05):
    """
    This class implements linear regression models
    Params:
    --------
    regularization - None for no regularization
                    'l2' for ridge regression
                    'l1' for lasso regression

    lam - lambda parameter for regularization in case of
        Lasso and Ridge

    learning_rate - learning rate for gradient descent algorithm,
                    used in case of Lasso

    tol - tolerance level for weight change in gradient descent
    """

    self.regularization = regularization
    self.lam = lam
    self.learning_rate = learning_rate
    self.tol = tol
    self.weights = None

  def fit(self, X, y):

    X = np.array(X)
    # first insert a column with all 1s in the beginning
    # hint: you can use the function np.insert
    X = np.insert(X, 0, [1] * X.shape[0], axis=1)

    if self.regularization is None:
      # the case when we don't apply regularization
      self.weights = np.linalg.inv(X.T @ X) @ (X.T @ y)
    elif self.regularization == 'l2':
      # the case of Ridge regression
      self.weights = np.linalg.inv(X.T @ X + self.lam * np.eye(X.shape[1])) @ (X.T @ y)
    elif self.regularization == 'l1':
      # in case of Lasso regression we use gradient descent
      # to find the optimal combination of weights that minimize the
      # objective function in this case (slide 37)

      # initialize random weights, for example normally distributed
      self.weights =  np.random.normal(0, 1, X.shape[1])

      converged = False
      # we can store the loss values to see how fast the algorithm converges
      self.loss = []
      # just a counter of algorithm steps
      i = 0
      while (not converged):
        i += 1
        # calculate the predictions in case of the weights in this stage
        y_pred = self.weights @ X.T
        # calculate the mean squared error (loss) for the predictions
        # obtained above
        self.loss.append(np.sum(np.power(y_pred - y, 2)) / len(y))
        # calculate the gradient of the objective function with respect to w
        # for the second component \sum|w_i| use np.sign(w_i) as it's derivative
        grad = X.T @ (y_pred - y) + self.lam * np.sign(self.weights)
        new_weights = self.weights - self.learning_rate * grad
        # check whether the weights have changed a lot after this iteration
        # compute the norm of difference between old and new weights
        # and compare with the pre-defined tolerance level, if the norm
        # is smaller than the tolerance level then we consider convergence
        # of the algorithm
        converged = np.linalg.norm(new_weights - self.weights) < self.tol
        self.weights = new_weights
      print(f'Converged in {i} steps')
  def predict(self, X):
    X = np.array(X)
    # don't forget to add the feature of 1s in the beginning
    X = np.insert(X, 0, [1] * X.shape[0], axis=1)
    # predict using the obtained weights
    return self.weights @ X.T


class DecisionNode:
  # we introduce a class for decision nodes in our tree
  # the object from this class will store all the necessary info
  # about the tree node such as
  # the index of the best feature, which resulted in better split (feature_id)
  # the threshold with which we compare the feature values (threshold)
  # if the node is a leaf, then the label will be stored in 'value' attribute
  # if the node is not a leaf, then it has
  # true_branch (condition is satisfied)
  # false_branch (condition is not satisfied) subtrees
  def __init__(self, feature_id=None, threshold=None,
               value=None, true_branch=None, false_branch=None):
    self.feature_id = feature_id
    self.threshold = threshold
    self.value = value
    self.true_branch = true_branch
    self.false_branch = false_branch

class RegressionTree:
  # This is the main class that recursively grows a decision tree and
  # predicts (again recursively) according to it
  def __init__(self, min_samples_split=2,
               min_impurity=1e-7, max_depth=float("inf")):
    # Minimum number of samples to perform spliting
    self.min_samples_split = min_samples_split

    # The minimum impurity to perform spliting
    self.min_impurity = min_impurity

    # The maximum depth to grow the tree until
    self.max_depth = max_depth

    # Root node in dec. tree
    self.root = None

  def calculate_purity_gain(self, y, y1, y2):
    # write the function to compute the purity gain after a split
    # use the formula (IG) from slide 26 from lectures
    # the fromula for gini index is similar, but using Gini_parent instead
    # the inputs are
    # y (labels before splitting)
    # y1 (labels from true branch after splitting)
    # y2 (labels from false branch after splitting)
    if len(y1) == len(y2) == 0:
      return 0

    y_mean = y.mean()
    purity = (1 / len(y)) * np.sum(y1 ** 2) * np.sum(y2 ** 2)
    for y_j in [y1, y2]:
      purity -= (len(y_j) / len(y)) * y_mean ** 2

    purity_parent = np.sum((y - y_mean) ** 2) / len(y)
    pure_gain = purity_parent - purity
    return pure_gain

  def divide_on_feature(self, X, y, feature_id, threshold):
    # This function divides the dataset into two parts according to the
    # logical operation that is performed on the feature with feature_id
    # comparing against threshold
    # you should consider 2 cases:
    # when the threshold is numeric, true cases will be feature >= threshold
    # and when it's not, true cases will be feature == threshold
    if isinstance(threshold, int) or isinstance(threshold, float):
      true_indices = X[:, feature_id] >= threshold
    else:
      true_indices = X[:, feature_id] == threshold

    X_1, y_1 = X[true_indices], y[true_indices]
    X_2, y_2 = X[~true_indices], y[~true_indices]

    return X_1, y_1, X_2, y_2

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    self.root = self.grow_tree(X, y)

  def grow_tree(self, X, y, current_depth=0):
    # this is the most important part
    # here we recursively grow the tree starting from the root
    # the depth of the tree is recorded so that we can later stop if we
    # reach the max_depth

    largest_purity_gain = 0 # initial small value for purity gain
    nr_samples, nr_features = np.shape(X)

    # checking if we have reached the pre-specified limits
    if nr_samples >= self.min_samples_split and current_depth <= self.max_depth:

      # go over the features to select the one that gives more purity
      for feature_id in range(nr_features):

        unique_values = np.unique(X[:, feature_id])

        # we iterate through all unique values of feature column and
        # calculate the impurity
        for threshold in unique_values:

          # Divide X and y according to the condition
          # if the feature value of X at index feature_id
          # meets the threshold
          X1, y1, X2, y2 = self.divide_on_feature(X, y, feature_id, threshold)

          # checking if we have samples in each subtree
          if len(X1) > 0 and len(X2) > 0:
            # calculate purity gain for the split
            purity_gain = self.calculate_purity_gain(y, y1, y2)

            # If this threshold resulted in a higher purity gain than
            # previously thresholds store the threshold value and the
            # corresponding feature index
            if purity_gain > largest_purity_gain:
              largest_purity_gain = purity_gain
              best_feature_id = feature_id
              best_threshold = threshold
              best_X1 = X1 # X of right subtree (true)
              best_y1 = y1 # y of right subtree (true)
              best_X2 = X2 # X of left subtree (true)
              best_y2 = y2 # y of left subtree (true)

    # if the resulting purity gain is good enough according our
    # pre-specified amount, then we continue growing subtrees using the
    # splitted dataset, we also increase the current_depth as
    # we go down the tree
    if largest_purity_gain > self.min_impurity:

      true_branch = self.grow_tree(best_X1,
                                   best_y1,
                                   current_depth + 1)

      false_branch = self.grow_tree(best_X2,
                                    best_y2,
                                    current_depth + 1)

      return DecisionNode(feature_id=best_feature_id,
                          threshold=best_threshold,
                          true_branch=true_branch,
                          false_branch=false_branch)

    # If none of the above conditions are met, then we have reached the
    # leaf of the tree  and we need to store the label
    leaf_value = y.mean()

    return DecisionNode(value=leaf_value)


  def predict_value(self, x, tree=None):
    # this is a helper function for the predict method
    # it recursively goes down the tree
    # x is one instance (row) of our test dataset

    # when we don't specify the tree, we start from the root
    if tree is None:
      tree = self.root

    # if we have reached the leaf, then we just take the value of the leaf
    # as prediction
    if tree.value is not None:
      return tree.value

    # we take the feature of the current node that we are on
    # to test whether our instance satisfies the condition
    feature_value = x[tree.feature_id]

    # determine if we will follow left (false) or right (true) branch
    # down the tree
    branch = tree.false_branch
    if isinstance(feature_value, int) or isinstance(feature_value, float):
      if feature_value >= tree.threshold:
        branch = tree.true_branch
    elif feature_value == tree.threshold:
      branch = tree.true_branch

    # continue going down the tree recursively through the chosen subtree
    # this function will finish when we reach the leaves
    return self.predict_value(x, branch)

  def predict(self, X):
    # Classify samples one by one and return the set of labels
    X = np.array(X)
    y_pred = [self.predict_value(instance) for instance in X]
    return np.array(y_pred)


class SVR:
  def __init__(self, epsilon=0.1, C=1, kernel_name='linear', power=2, gamma=None, coef=2):
    self.C = C
    self.power = power # degree of the polynomial kernel (d in the slides)
    self.gamma = gamma # Kernel coefficient for "rbf" and "poly"
    self.coef = coef # coefficent of the polynomial kernel (r in the slides)
    self.kernel_name = kernel_name  # implement for 'linear', 'poly' and 'rbf'
    self.epsilon = epsilon
    self.kernel = None
    self.alphas_minus = None
    self.support_vectors = None
    self.support_vector_labels = None
    self.t = None

  def get_kernel(self, kernel_name):
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
    Q = kernel_matrix
    e = np.ones(nr_samples)
    a = cp.Variable(nr_samples)
    a_ = cp.Variable(nr_samples)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(a - a_, Q) + self.epsilon * e.T @ (a + a_) - y.T @ (a - a_)),
                    [0 <= a,
                     a <= self.C,
                     0 <= a_,
                     a_ <= self.C,
                     e.T @ (a - a_) == 0])

    # Solve the quadratic optimization problem using cvxopt
    prob.solve()

    alphas = a.value
    alphas_ = a_.value

    # first get indexes of non-zero lagr. multipiers
    idx = (alphas > 1e-7) & (alphas_ > 1e-7)
    ind = np.arange(len(alphas))[idx]

    # get the corresponding lagr. multipliers (non-zero alphas)
    self.alphas_minus = alphas[idx] - alphas_[idx]

    # get the support vectors
    self.support_vectors = X[idx]

    # get the corresponding labels
    self.support_vector_labels = y[idx]

    # Calculate intercept (t) with first support vector
    self.t = self.support_vector_labels[0] - self.epsilon
    for i in range(len(self.alphas_minus)):
      self.t -= self.alphas_minus[i] * kernel_matrix[ind[i], 0]

  def predict(self, X):
    y_pred = []
    for instance in np.array(X):
      prediction = 0
      # determine the label of the given instance by the support vectors
      for i in range(len(self.alphas_minus)):
        prediction += self.alphas_minus[i] * self.kernel(self.support_vectors[i], instance)
      prediction += self.t
      y_pred.append(prediction)
    return np.array(y_pred)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class LogisticRegression:
   def __init__(self, learning_rate=1e-3, nr_iterations=10, batch_size=64):
    self.learning_rate = learning_rate
    self.nr_iterations = nr_iterations
    self.batch_size = batch_size
    self.weights = None

   def fit(self, X, y):
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    if not isinstance(y, np.ndarray):
      y = np.array(y)
    # first insert a column with all 1s in the beginning
    # hint: you can use the function np.insert
    X = np.insert(X, 0, [1] * X.shape[0], axis=1)

    # initialize random weights, for example normally distributed
    self.weights =  np.random.normal(0, 1, X.shape[1])

    # we can store the loss values to see how fast the algorithm converges
    self.loss = []
    for _ in range(self.nr_iterations):
      indx = np.random.choice(np.arange(len(y)), size=self.batch_size)
      X_batch = X[indx, :]
      y_batch = y[indx]
      # calculate the predictions in case of the weights in this stage
      y_pred = np.array([sigmoid(self.weights.T @ X_batch[i]) for i in range(self.batch_size)])
      # calculate the mean squared error (loss) for the predictions
      # obtained above
      self.loss.append(np.sum(np.power(y_pred - y_batch, 2)) / len(y_batch))
      # calculate the gradient of the objective function with respect to w
      grad = - X_batch[0] * (y_batch[0] - y_pred[0])
      for i in range(1, self.batch_size):
        grad -= X_batch[i] * (y_batch[i] - y_pred[i])
      new_weights = self.weights - self.learning_rate * grad
      self.weights = new_weights

   def predict(self, X):
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    # don't forget to add the feature of 1s in the beginning
    X = np.insert(X, 0, [1] * X.shape[0], axis=1)
    # predict using the obtained weights
    return np.array([sigmoid(self.weights.T @ X[i]) for i in range(len(X))])



class SoftmaxClassifier:
  def __init__(self, learning_rate=1e-3, nr_iterations=10,
   batch_size=64):
    self.learning_rate = learning_rate # learning rate for the GD
    self.nr_iterations = nr_iterations # number of iterations for GD
    self.batch_size = batch_size  # batch size for the GD
    self.bar = progressbar.ProgressBar(widgets=widgets)
    self.W = None  # weight matrix
    self.classes = None

  """ This softmax is not working for large numbers """
  # @staticmethod
  # def softmax(z):
  #   # write the softmax function as it is writen in the notebook
  #   return np.exp(z) / np.sum(np.exp(z[1:]), axis=0)

  @staticmethod
  def softmax(x):
    axis = len(x.shape) - 1
    # make every value 0 or below, as exp(0) won't overflow
    xrel = x - x.max(axis=axis, keepdims=True)

    exp_xrel = np.exp(xrel)
    return exp_xrel / exp_xrel.sum(axis=axis, keepdims=True)

  @staticmethod
  def one_hot_encoding(a):
    new_arr = []
    b = np.unique(a)
    for i in range(len(a)):
      new_arr.append((b == a[i]).astype(int))
    return np.asarray(new_arr)

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    self.classes = np.unique(y)

    # insert 1s as the first feature
    X = np.insert(X, 0, 1, axis=1)

    nr_samples, nr_features = X.shape
    nr_classes = len(np.unique(y))

    # transform y into a one-hot encoded matrix and denote it with Y
    Y = self.one_hot_encoding(y)

    # intitialize a random matrix of size (n x m) for the weights
    if self.W is None:
        self.W = 0.0001 * np.random.randn(nr_features, nr_classes)

    self.loss = []
    for i in self.bar(range(self.nr_iterations)):
      # select samples from the data according to the batch size
      # Hint: you can use np.random.choice to select indices
      indx = np.random.choice(np.arange(len(y)), size=self.batch_size)
      X_batch = X[indx, :] # n x m matrix, nr_sample := n, nr_features := m
      Y_batch = Y[indx, :] # n x k matrix, nr_classes := k

      # get the probability matrix (p) using X_batch and the current W matrix
      # Hint: you need to apply softmax function to get probabilities
      # it will be a matrix of size n x k
      p = self.softmax(np.dot(X_batch, self.W))

      # get the loss (matrix) using the log(p) and Y_batch
      # think about the loss function formula (L) as a dot product
      # it will be a matrix of size n x n,
      # where the diagonals are the losses per sample point
      loss_matrix = np.dot(np.ma.log(p), Y_batch.T)

      # get the average loss across the batch
      loss = -np.mean(loss_matrix.diagonal())
      self.loss.append(loss)

      # compute the gradient using the last formula in the notebook
      # don't forget to normalize the gradient with the batch size
      # since we took the normalized cross-entropy (mean instead of sum)
      # think about the gradient as a dot product
      # the result should be an m x k matrix (the same size as W)
      gradient = (1 / self.batch_size) * np.dot(X_batch.T, (p - Y_batch))

      self.W -= self.learning_rate * gradient

  def predict(self, X):
    X = np.array(X)
    X = np.insert(X, 0, 1, axis=1)

    # use the weight matrix W to obtain the probabilities with softmax
    prob = self.softmax(np.dot(X, self.W))
    # get the index of the highest probability per row as the prediction
    # you may want to use np.argmax here
    idx = np.argmax(prob, axis=1)
    return self.classes[idx]
