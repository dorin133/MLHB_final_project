# Ranker
## YOUR SOLUTION
import sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score
import itertools
from sklearn import svm, linear_model

# def transform_pairwise(X,y):
#     # dims
#     s = np.unique(y[:,1]).shape[0]
#     n = y.shape[0]/s
#     q = X.shape[1]
#     k = n-1
#
#     binary_labels = y[:,0]
#     Yj = X[binary_labels == 1]
#     Yj_repeat = np.repeat(Yj, repeats=k, axis=0)
#     Y_not_j = X[binary_labels == 0]
#
#     np.random.seed(0)
#     y_new = np.random.choice([-1,1], int(s*k), p=[0.5,0.5])
#     X_new = (Yj_repeat-Y_not_j)*y_new.reshape(int(s*k),-1)
#
#     return n, X_new,y_new
#
# class MyRanker(sklearn.svm.SVC):
#     def fit(self, X, y):
#         amount_of_items_per_set, X_new, y_new = transform_pairwise(X, y)
#         self.n = int(amount_of_items_per_set)
#         return super().fit(X_new, y_new)
#

def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []

    # dims
    s = np.unique(y[:,1]).shape[0]
    n = y.shape[0]/s
    q = X.shape[1]
    k = n-1

    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return n, np.asarray(X_new), np.asarray(y_new).ravel()


class MyRanker(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        amount_of_items_per_set, X_trans, y_trans = transform_pairwise(X, y)
        self.n = int(amount_of_items_per_set)
        super(MyRanker, self).fit(X_trans, y_trans)
        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_.ravel())

    def predict(self, X):
        y_pred = X @ self.coef_.T
        y_pred_reshaped = y_pred.reshape(-1, int(self.n))
        y_argmax = y_pred_reshaped.argmax(axis=1)
        onehot = np.zeros((y_argmax.size, self.n))
        onehot[np.arange(y_argmax.size), y_argmax] = 1
        onehot = onehot.reshape(-1)
        return onehot

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(MyRanker, self).predict(X_trans) == y_trans)

