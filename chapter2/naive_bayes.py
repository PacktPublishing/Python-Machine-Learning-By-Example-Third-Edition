'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 2 Building A Movie Recommendation Engine with Naive Bayes
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import numpy as np


def get_label_indices(labels):
    """
    Group samples based on their labels and return indices
    @param labels: list of labels
    @return: dict, {class1: [indices], class2: [indices]}
    """
    from collections import defaultdict
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices

def get_prior(label_indices):
    """
    Compute prior based on training samples
    @param label_indices: grouped sample indices by class
    @return: dictionary, with class label as key, corresponding prior as the value
    """
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior

def get_likelihood(features, label_indices, smoothing=0):
    """
    Compute likelihood based on training samples
    @param features: matrix of features
    @param label_indices: grouped sample indices by class
    @param smoothing: integer, additive smoothing parameter
    @return: dictionary, with class as key, corresponding conditional probability P(feature|class) vector as value
    """
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood

def get_posterior(X, prior, likelihood):
    """
    Compute posterior of testing samples, based on prior and likelihood
    @param X: testing samples
    @param prior: dictionary, with class label as key, corresponding prior as the value
    @param likelihood: dictionary, with class label as key, corresponding conditional probability vector as value
    @return: dictionary, with class label as key, corresponding posterior as value
    """
    posteriors = []
    for x in X:
        # posterior is proportional to prior * likelihood
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
        # normalize so that all sums up to 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors



if __name__ == '__main__':

    X_train = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0]])

    Y_train = ['Y', 'N', 'Y', 'Y']

    X_test = np.array([[1, 1, 0]])

    label_indices = get_label_indices(Y_train)
    print('label_indices:\n', label_indices)

    prior = get_prior(label_indices)
    print('Prior:', prior)

    smoothing = 1
    likelihood = get_likelihood(X_train, label_indices, smoothing)
    print('Likelihood:\n', likelihood)


    posterior = get_posterior(X_test, prior, likelihood)
    print('Posterior:\n', posterior)


    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB(alpha=1.0, fit_prior=True)
    clf.fit(X_train, Y_train)

    pred_prob = clf.predict_proba(X_test)
    print('[scikit-learn] Predicted probabilities:\n', pred_prob)

    pred = clf.predict(X_test)
    print('[scikit-learn] Prediction:', pred)