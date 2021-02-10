import numpy as np
import progressbar

# you need Regresssion trees, so use either your implementation or sklearn's
from sklearn.tree import DecisionTreeRegressor

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

# Instead of using a Decision Tree with one level
# we can create another object for Decision Stump
# which will work faster since it will not compute impurity
# to decide on which feature to make a split

# after implementing this version, create a different Adaboost
# that uses decision trees with one level and check that it is
# more inefficient compared to the below implementation.

class DecisionStump():
  def __init__(self):
    # we will use this attribute to convert the predictions
    # in case the error > 50%
    self.flip = 1
    # the feature index on which the split was made
    self.feature_index = None
    # the threshold based on which the split was made
    self.threshold = None
    # the confidence of the model (see the pseudocode from the lecture slides)
    self.alpha = None

class Adaboost():
  # this implementation supports only -1,1 label encoding
  def __init__(self, nr_estimators=5):
    # number of weak learners (Decision Stumps) to use
    self.nr_estimators = nr_estimators
    self.progressbar = progressbar.ProgressBar(widgets=widgets)

  def fit(self, X, y):
    nr_samples, nr_features = np.shape(X)

    # initialize the uniform weights for each training instance
    w = np.full(X.shape[0], 1 / X.shape[0])

    self.models = []
    for i in self.progressbar(range(self.nr_estimators)):
      model = DecisionStump()

      # we set the initial error very high in order to select
      # the model with lower error
      min_error = 1

      # we go over each feature as in case of decision tree
      # to decide which split leads to a smaller error
      # note that here we don't care about the impurity
      # even if we find a model with 90% error, we will flip the
      # sign of the predictions and will make it a model with 10% error
      for feature_id in range(nr_features):
        unique_values = np.unique(X[:, feature_id])
        thresholds = (unique_values[1:] + unique_values[:-1]) / 2
        for threshold in thresholds:
          # setting an intial value for the flip
          flip = 1
          # setting all the predictions as 1
          prediction = np.ones(nr_samples)
          # if the feature has values less than the fixed threshold
          # then it's prediction should be manually put as -1
          prediction[X[:, feature_id] < threshold] = -1

          # compute the weighted error (epsilon_t) for the resulting prediction
          error = np.sum(w[prediction != y])

          # if the model is worse than random guessing
          # then we need to set the flip variable to -1
          # so that we can use it later, we also modify the error
          # accordingly
          if error > 0.5:
            error = 1 - error
            flip = -1

          # if this feature and threshold were the one giving
          # the smallest error, then we store it's info in the 'model' object
          if error < min_error:
            model.flip = flip
            model.threshold = threshold
            model.feature_index = feature_id
            min_error = error

      # compute alpha based on the error of the 'best' decision stump
      model.alpha = 1/2 * np.log((1-min_error)/min_error)

      # obtain the predictions from the chosen decision stump
      # using the info stored in the 'model' object
      # don't forget about the flip if necessary
      prediction = np.ones(nr_samples)
      # if the feature has values less than the fixed threshold
      # then it's prediction should be manually put as -1
      if model.flip == 1:
        prediction[X[:, model.feature_index] < model.threshold] = -1
      else:
        prediction[X[:, model.feature_index] >= model.threshold] = -1

      # compute the weights and normalize them
      w *= np.exp(-model.alpha * y * prediction)
      w /= np.sum(w)

      # store the decision stump of the current iteration for later
      self.models.append(model)

  def predict(self, X):
    nr_samples = np.shape(X)[0]
    y_pred = np.zeros(nr_samples)

    # for each instance in X you should obtain the 'prediction'
    # from each decision stump (not forgetting about the flip variable)
    # then take the sum of
    # all the individual predictions times their weights (alpha)
    # if the resulting amount is bigger than 0 then predict 1, otherwise -1
    for model in self.models:
      prediction = np.ones(nr_samples)
      if model.flip == 1:
        prediction[X[:, model.feature_index] < model.threshold] = -1
      else:
        prediction[X[:, model.feature_index] >= model.threshold] = -1
      prediction *= model.alpha
      y_pred += prediction
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1

    return y_pred


class GradientBoostingRegressor:
  def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
               min_impurity=1e-7, max_depth=4):
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.min_samples_split = min_samples_split
    self.min_impurity = min_impurity
    self.max_depth = max_depth
    self.bar = progressbar.ProgressBar(widgets=widgets)

    # write the square loss function as in the lectures
    def square_loss(y, y_pred): return np.power(y - y_pred, 2) / 2

    # write the gradient of square loss as in the lectures
    def square_loss_gradient(y, y_pred): return y_pred - y

    self.loss = square_loss
    self.loss_gradient = square_loss_gradient

  def fit(self, X, y):
    self.trees = [] # we will store the regression trees per iteration
    self.train_loss = [] # we will store the loss values per iteration

    # initialize the predictions (f(x) in the lectures)
    # with the mean values of y
    # hint: you may want to use the np.full function here
    self.mean_y = np.mean(y)
    y_pred = np.full(y.shape[0], self.mean_y)
    for i in self.bar(range(self.n_estimators)):
      tree = DecisionTreeRegressor(
              min_samples_split=self.min_samples_split,
              min_impurity_decrease=self.min_impurity,
              max_depth=self.max_depth) # this is h(x) from our lectures
      # get the loss when comparing y_pred with true y
      # and store the values in self.train_loss
      loss = self.loss(y, y_pred)
      self.train_loss.append(loss)

      # get the pseudo residuals
      residuals = -self.loss_gradient(y, y_pred)

      tree.fit(X, residuals) # fit the tree on the residuals
      # update the predictions y_pred using the tree predictions on X
      y_pred += self.learning_rate * tree.predict(X)

      self.trees.append(tree) # store the tree model

  def predict(self, X):
    # start with initial predictions as vector of
    # the mean values of y_train (self.mean_y)
    y_pred = np.full(X.shape[0], self.mean_y)
    # iterate over the regression trees and apply the same gradient updates
    # as in the fitting process, but using test instances
    for tree in self.trees:
      y_pred += self.learning_rate * tree.predict(X)
    return y_pred


class GradientBoostingClassifier:
  def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
               min_impurity=1e-7, max_depth=4):
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.min_samples_split = min_samples_split
    self.min_impurity = min_impurity
    self.max_depth = max_depth
    self.bar = progressbar.ProgressBar(widgets=widgets)

    # write the square loss function as in the lectures
    def square_loss(y, y_pred): return -np.sum(y * np.log(y_pred))

    # write the gradient of square loss as in the lectures
    def square_loss_gradient(y, y_pred): return y_pred - y

    self.loss = square_loss
    self.loss_gradient = square_loss_gradient

  @staticmethod
  def one_hot_encoding(a):
    new_arr = []
    b = np.unique(a)
    for i in range(len(a)):
      new_arr.append((b == a[i]).astype(int))
    return np.asarray(new_arr)

  @staticmethod
  def softmax(x):
    axis = len(x.shape) - 1
    xrel = x - x.max(axis=axis, keepdims=True)

    exp_xrel = np.exp(xrel)
    return exp_xrel / exp_xrel.sum(axis=axis, keepdims=True)

  def fit(self, X, y):
    self.trees = [] # we will store the regression trees per iteration
    self.train_loss = [] # we will store the loss values per iteration

    Y = self.one_hot_encoding(y)
    self.K = Y.shape[1]

    # initialize the predictions (f(x) in the lectures)
    y_pred = np.full(Y.shape, 1 / self.K)
    for i in self.bar(range(self.n_estimators)):
      # get the loss when comparing y_pred with true y
      # and store the values in self.train_loss
      loss = self.loss(Y, y_pred)
      self.train_loss.append(loss)

      # get the pseudo residuals
      residuals = np.empty(Y.shape)
      for i in range(Y.shape[0]):
        residuals[i] = -self.loss_gradient(Y[i], y_pred[i])

      trees = []
      for j in range(self.K):
        tree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity,
                max_depth=self.max_depth)
        tree.fit(X, residuals[:, j]) # fit the tree on the residuals
        # update the predictions y_pred using the tree predictions on X
        y_pred[:, j] += self.learning_rate * self.softmax(tree.predict(X))
        trees.append(tree)

      self.trees.append(trees)

  def predict(self, X):
    # start with initial predictions as vector of
    # the mean values of y_train (self.mean_y)
    y_pred = np.full((X.shape[0], self.K), 1 / self.K)
    # iterate over the regression trees and apply the same gradient updates
    # as in the fitting process, but using test instances
    for i in range(len(self.trees)):
      for j in range(len(self.trees[i])):
        tree = self.trees[i][j]
        y_pred[:, j] += self.learning_rate * self.softmax(tree.predict(X))
    return y_pred.argmax(axis=1)
