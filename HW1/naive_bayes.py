import numpy as np

class MyNaiveBayes:
  def __init__(self, smoothing=False):
    # initialize Laplace smoothing parameter
    self.smoothing = smoothing

  def fit(self, X_train, y_train):
    # use this method to learn the model
    # if you feel it is easier to calculate priors
    # and likelihoods at the same time
    # then feel free to change this method
    self.X_train = X_train
    self.y_train = y_train
    self.priors = self.calculate_priors()
    self.likelihoods = self.calculate_likelihoods()

  def predict(self, X_test):
    # recall: posterior is P(label_i|feature_j)
    # hint: Posterior probability is a matrix of size
    #       m*n (m samples and n labels)
    #       our prediction for each instance in data is the class that
    #       has the highest posterior probability.
    #       You do not need to normalize your posterior,
    #       meaning that for classification, prior and likelihood are enough
    #       and there is no need to divide by evidence. Think why!
    # return: a list of class labels (predicted)

    prediction = []
    for _, row in X_test.iterrows():
      label_probabilities = []
      for label in self.likelihoods:
        likelihood = self.likelihoods[label]
        label_probability = self.priors[label] * np.prod([likelihood[i][row[i]] for i in range(len(likelihood))])
        label_probabilities.append((label, label_probability))
      predicted_label = max(label_probabilities, key=lambda arr: arr[1])[0]
      prediction.append(predicted_label)
    return np.array(prediction)

  def calculate_priors(self):
    # recall: prior is P(label=l_i)
    # hint: store priors in a pandas Series or a list

    a = self.y_train.value_counts()
    b = len(self.y_train)
    if self.smoothing:
      a += 1
      b += len(self.y_train.unique())
    priors = a / b
    return priors

  def calculate_likelihoods(self):
    # recall: likelihood is P(feature=f_j|label=l_i)
    # hint: store likelihoods in a data structure like dictionary:
    #        feature_j = [likelihood_k]
    #        likelihoods = {label_i: [feature_j]}
    #       Where j implies iteration over features, and
    #             k implies iteration over different values of feature j.
    #       Also, i implies iteration over different values of label.
    #       Likelihoods, is then a dictionary that maps different label
    #       values to its corresponding likelihoods with respect to feature
    #       values (list of lists).
    #
    #       NB: The above pseudocode is for the purpose of understanding
    #           the logic, but it could also be implemented as it is.
    #           You are free to use any other data structure
    #           or way that is convenient to you!
    #
    #       More Coding Hints: You are encouraged to use Pandas as much as
    #       possible for all these parts as it comes with flexible and
    #       convenient indexing features which makes the task easier.

    likelihoods = {}

    unique_labels = self.y_train.unique()
    unique_labels_count = len(unique_labels)
    col_number = self.X_train.shape[1]
    for i in range(unique_labels_count):
      label = unique_labels[i]
      label_data = self.X_train[self.y_train == label]
      label_data_count = len(label_data)
      likelihoods[label] = []
      for j in range(col_number):
        col_values = label_data.iloc[:, j]
        a = col_values.value_counts()
        b = label_data_count
        if self.smoothing:
          a += 1
          b += unique_labels_count
        likelihoods[label].append(a / b)

    return likelihoods
