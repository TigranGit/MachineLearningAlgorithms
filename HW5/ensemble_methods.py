import numpy as np
from scipy import stats

# if you want to use your own Decision Tree implementation for Random Forest
from decision_tree import DTClassifier

# something useful for tracking algorithm's iterations
import progressbar

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

def get_bootstrap_samples(X, y, nr_bootstraps, nr_samples=None):
  # this function is for getting bootstrap samples with replacement
  # from the initial dataset (X, y)
  # nr_bootstraps is the number of bootstraps needed
  # nr_samples is the number of data points to sample each time
  # it should be the size of X, if nr_samples is not provided
  # Hint: you may need np.random.choice function somewhere in this function
  if nr_samples is None:
    nr_samples = X.shape[0]
  bootstrap_samples = []
  for i in range(nr_bootstraps):
    indexes = np.arange(X.shape[0])
    random_indexes = np.random.choice(indexes, size=nr_samples, replace=True)
    bootstrap_samples.append((X[random_indexes].copy(), y[random_indexes].copy()))
  return bootstrap_samples

class Bagging:
  def __init__(self, base_estimator, nr_estimators=10):
    # number of models in the ensemble
    self.nr_estimators = nr_estimators
    self.progressbar = progressbar.ProgressBar(widgets=widgets)
    # this can be any object that has 'fit', 'predict' methods
    self.base_estimator = base_estimator

  def fit(self, X, y):
    # this method will fit a separate model (self.base_estimator)
    # on each bootstrap sample and each model should be stored
    # in order to use it in 'predict' method
    X = np.array(X)
    y = np.array(y)
    bootstrap_samples = get_bootstrap_samples(X, y,
                                              nr_bootstraps=self.nr_estimators)
    self.models = []
    for i in self.progressbar(range(self.nr_estimators)):
      X_test, y_test = bootstrap_samples[i]
      model = self.base_estimator()
      model.fit(X_test, y_test)
      self.models.append(model)

  def predict(self, X):
    # this method will predict the labels for a given test dataset
    # get the majority 'vote' for each test instance from the ensemble
    # Hint: you may want to use 'mode' method from scipy.stats
    y_predictions = np.array([model.predict(X) for model in self.models])
    y_preds = stats.mode(y_predictions)[0][0]
    return y_preds

class RandomForest:
  def __init__(self, nr_estimators=10, max_features=None, min_samples_split=2,
                min_gain=0, max_depth=float("inf")):
    # number of trees in the forest
    self.nr_estimators = nr_estimators

    # this is the number of features to use for each tree
    # if not specified this should be set to sqrt(initial number of features)
    self.max_features = max_features

    # the rest is for decision tree
    self.min_samples_split = min_samples_split
    self.min_gain = min_gain
    self.max_depth = max_depth
    self.progressbar = progressbar.ProgressBar(widgets=widgets)

  def fit(self, X, y):
    # this method will fit a separate tree
    # on each bootstrap sample and subset of features
    # each tree should be stored
    # in order to use it in 'predict' method

    X = np.array(X)
    y = np.array(y)
    bootstrap_samples = get_bootstrap_samples(X, y,
                                              self.nr_estimators)

    self.trees = []
    for i in self.progressbar(range(self.nr_estimators)):
      # you can modify the code to use sklearn's decision tree
      # if you don't want to use your implementation
      tree = DTClassifier(
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_gain,
                max_depth=self.max_depth)
      X_boot, y_boot = bootstrap_samples[i]

      count = self.max_features if self.max_features else int(np.sqrt(X.shape[1]))
      idx = np.random.choice(X.shape[1], size=count, replace=False)

      # we need to keep the indices of the features used for this tree
      tree.feature_indices = idx
      tree.fit(X_boot[:, idx], y_boot)
      self.trees.append(tree)

  def predict(self, X):
    # this method will predict the labels for a given test dataset
    # get the majority 'vote' for each test instance from the forest
    # Hint: you may want to use 'mode' method from scipy.stats
    # besides the individual trees, you will also need the feature indices
    # it was trained on
    y_predictions = np.array([tree.predict(X[:, tree.feature_indices]) for tree in self.trees])
    y_preds = stats.mode(y_predictions)[0][0]
    return y_preds

class WeightedVoting:
  def __init__(self, estimators, num_folds=3):
    # list of classifier objects
    self.estimators = estimators
    self.nr_estimators = len(estimators)
    self.weights = None
    self.num_folds = num_folds

  def get_weights(self, X, y):
    # This method is for deriving the weights of each individual classifier
    # using cross-validation as described in the lecture slides
    # the output should be an array of weights
    weights = np.empty(self.nr_estimators)
    X_folds = np.array_split(X, self.num_folds)
    y_folds = np.array_split(y, self.num_folds)
    X_folds= [el for el in X_folds if el.size > 0]
    y_folds= [el for el in y_folds if el.size > 0]
    for i_estimator, estimator in enumerate(self.estimators):
      avg_acc = 0
      for i in range(self.num_folds):
        X_train, X_val = np.concatenate(X_folds[:i] + X_folds[i+1:]), X_folds[i]
        y_train, y_val = np.concatenate(y_folds[:i] + y_folds[i+1:]), y_folds[i]
        estimator.fit(X_train, y_train)
        prediction = estimator.predict(X_val)
        acc = (prediction == y_val).sum() / len(y_val)
        avg_acc += acc
      avg_acc /= self.num_folds
      weights[i_estimator] = avg_acc
    weights /= weights.sum()
    return weights

  def fit(self, X, y):
    # Train the individual models on the whole training dataset
    # and update self.estimators accordingly in order to use them for prediction
    self.weights = self.get_weights(X, y)
    for estimetor in self.estimators:
      estimetor.fit(X, y)

  def predict(self, X):
    # Use the fitted individual models and their weights to perform prediction
    # This link may be useful
    # https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
    predictions_proba = self.estimators[0].predict_proba(X) * self.weights[0]
    for estimator, weight in zip(self.estimators[1:], self.weights[1:]):
      predictions_proba += estimator.predict_proba(X) * weight
    predictions = predictions_proba.argmax(axis=1)
    return predictions


class Stacking:
  def __init__(self, estimators, final_estimator, meta_features='class', cv=False, k=None):
    # list of classifier objects
    self.estimators = estimators
    # classifier for the meta-model
    self.final_estimator = final_estimator
    self.nr_estimators = len(estimators)
    # meta-features (input) of the meta-model, this should take to values either
    # 'class' if we take the predicted labels or
    # 'prob' if we take the class probabilities from the individual models.

    # In case of 'prob' you need to use 'predict_proba' method on sklearn classifiers
    # and need to discard one of the probability values. For example, if the task is a 2 class classification problem,
    # then each individual model's predict_proba method will return a vector of 2 values for each class's probability
    # and we can discard one of those values because it is the complement of the other class. ([p, q], where q = 1-p)
    # In case we have a m-class classification problem and T individual models, then the input for the meta-model will be
    # T * (m-1) dimensional vector, since each model will give m-1 probability values.

    # In case of 'class', the input for the meta-model will be a T dimensional vector.
    self.meta_features = meta_features
    # boolean specifying whether to use cross validation for deriving the meta-features or not, as described in the lecture slides
    self.cv = cv
    # number of folds of cross validation
    self.k = k

  def _predict(self, estimator, X):
    if self.meta_features == 'class':
          prediction = estimator.predict(X)
    elif self.meta_features == 'prob':
      prediction = estimator.predict_proba(X)
      prediction = prediction[:, 1:]
      if prediction.shape[1] == 1:
        prediction = prediction.ravel()
    else:
      raise ValueError('meta_features is invalid')
    return prediction

  def get_predictions(self, X, y=None, fit=False):
    predictions = []
    if not self.cv or not fit:
      for estimator in self.estimators:
        if fit:
          estimator.fit(X, y)
        prediction = self._predict(estimator, X)
        predictions.append(prediction)
    else:
      X_folds = np.array_split(X, self.k)
      y_folds = np.array_split(y, self.k)
      X_folds= [el for el in X_folds if el.size > 0]
      y_folds= [el for el in y_folds if el.size > 0]
      for estimator in self.estimators:
        predictions_estimator = []
        for i in range(self.k):
          X_train, X_val = np.concatenate(X_folds[:i] + X_folds[i+1:]), X_folds[i]
          y_train, y_val = np.concatenate(y_folds[:i] + y_folds[i+1:]), y_folds[i]
          estimator.fit(X_train, y_train)
          prediction = self._predict(estimator, X_val)
          predictions_estimator.append(prediction)
        predictions.append(np.concatenate(np.array(predictions_estimator)))
    return np.array(predictions).T

  def fit(self, X, y):
    # Derive the meta-features and train the meta-model on it
    predictions = self.get_predictions(X, y, fit=True)
    # print(predictions)
    self.final_estimator.fit(predictions, y)

  def predict(self, X):
    # Get the predictions of the individual models and provide them as inputs to the meta-model
    predictions = self.get_predictions(X)
    result = self.final_estimator.predict(predictions)
    return result
