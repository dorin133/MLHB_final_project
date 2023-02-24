# Ranker
## YOUR SOLUTION
import sklearn
import numpy as np
import pandas as pd

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


import itertools
from sklearn import svm, linear_model


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

    def recommend_items_aux(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        The item is given such that items ranked on top have are
        predicted a higher ordering (i.e. 0 means is the last item
        and n_samples would be the item ranked on top).
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.ravel()))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def recommend_items_ranker(self, X_test, num_items_to_recom=3):
        slate_grouped_X_test = X_test.groupby('slate_id')
        # print(slate_grouped_X_test)

        recommendations_indices = pd.DataFrame(columns=['slate_id', 'item_index'])
        recommendations_features = pd.DataFrame(columns=['slate_id', 'x_0', 'x_1', 'x_2', 'x_3', 'x_4'])
        for slate_id, slate_items in slate_grouped_X_test:

            order_indices = self.recommend_items_aux(slate_items[['x_0', 'x_1', 'x_2', 'x_3', 'x_4']])
            # use iloc to select the rows in the desired order
            recom_slate_items_features = slate_items.iloc[order_indices].head(num_items_to_recom)
            recom_slate_indices = order_indices[:num_items_to_recom]
            slate_recom_indices = pd.DataFrame(
                {'slate_id': [slate_id] * num_items_to_recom, 'item_index': recom_slate_indices})
            recommendations_indices = pd.concat([recommendations_indices, slate_recom_indices])
            recommendations_features = pd.concat([recommendations_features, recom_slate_items_features])
            # print(recommendations))
        return recommendations_features, recommendations_indices

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


