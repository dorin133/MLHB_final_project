# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score

import numpy as np
import pandas as pd


def train_models_per_user(X_train, X_test, y_train, y_test, model, user_types = ['Rational','Compromise', 'Similarity', 'Attraction']):

  results = dict((user_type,{'accuracy': 0, 'precision': 0}) for user_type in user_types)

  _X_train = X_train[['x_0', 'x_1', 'x_2', 'x_3','x_4']]
  _X_test =  X_test[['x_0', 'x_1', 'x_2', 'x_3','x_4']]

  models = {}
  for user_type in user_types:
    _y_train = y_train[user_type].to_numpy()
    _y_test = y_test[user_type].to_numpy()

    clf = model.fit(_X_train, _y_train)
    models[user_type] = clf
    y_pred = clf.predict(_X_test)
    # print(f'y_pred = {y_pred}')
    precision = precision_score(_y_test,y_pred, zero_division=0)
    accuracy = accuracy_score(_y_test,y_pred)
    results[user_type]['accuracy'] = accuracy
    results[user_type]['precision'] = precision

  return models, results

def recommend_items_wraper(X_train, X_test, y_train, y_test, model_dict, num_items_to_recom, user_types = ['Rational', 'Compromise', 'Similarity', 'Attraction']):
    recommendations_features_curr_model = {}
    # non-sklearn wrraped models:
    if model_dict['model_name'] == "pairwise_ranker":
        models_per_user, _ = train_pairwise_ranker_models(X_train, X_test, y_train, y_test, model_dict['model_init'])
        for user_name in user_types:
            recommendations_features_curr_model[user_name], _ = recommend_items_ranker(models_per_user[user_name], X_test, num_items_to_recom)
    else:
        models_per_user, _ = train_models_per_user(X_train, X_test, y_train, y_test, model_dict['model_init'])
        for user_name in user_types:
            recommendations_features_curr_model[user_name], _ = recommend_items(models_per_user[user_name], X_test, num_items_to_recom)

    return recommendations_features_curr_model

def recommend_items(model, X_test, num_items_to_recom = 3):
  slate_grouped_X_test = X_test.groupby('slate_id')
  # print(slate_grouped_X_test)

  recommendations_indices  = pd.DataFrame(columns = ['slate_id','item_index'])
  recommendations_features = pd.DataFrame(columns = ['slate_id', 'x_0', 'x_1', 'x_2', 'x_3','x_4'])
  for slate_id, slate_items in slate_grouped_X_test:
    # print(slate_items)
    pred_scores = model.predict_proba(slate_items[['x_0', 'x_1', 'x_2', 'x_3','x_4']])[:, 0]  #gives a score in [0, 1]
    # print(pred_scores)
    # get the indices that would sort the vector in descending order
    order_indices = np.argsort(pred_scores)[::-1]
    # use iloc to select the rows in the desired order
    recom_slate_items_features = slate_items.iloc[order_indices].head(num_items_to_recom)
    recom_slate_indices = order_indices[:num_items_to_recom]
    slate_recom_indices = pd.DataFrame({'slate_id': [slate_id] * num_items_to_recom, 'item_index':recom_slate_indices})
    recommendations_indices = pd.concat([recommendations_indices, slate_recom_indices])
    recommendations_features = pd.concat([recommendations_features, recom_slate_items_features])
    # print(recommendations))
  return recommendations_features, recommendations_indices


def recommend_items_ranker_aux(model, X):
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
    if hasattr(model, 'coef_'):
        return np.argsort(np.dot(X, model.coef_.ravel()))
    else:
        raise ValueError("Must call fit() prior to predict()")

def recommend_items_ranker(model, X_test, num_items_to_recom=3):
    slate_grouped_X_test = X_test.groupby('slate_id')
    # print(slate_grouped_X_test)

    recommendations_indices = pd.DataFrame(columns=['slate_id', 'item_index'])
    recommendations_features = pd.DataFrame(columns=['slate_id', 'x_0', 'x_1', 'x_2', 'x_3', 'x_4'])
    for slate_id, slate_items in slate_grouped_X_test:

        order_indices = recommend_items_ranker_aux(model, slate_items[['x_0', 'x_1', 'x_2', 'x_3', 'x_4']])
        # use iloc to select the rows in the desired order
        recom_slate_items_features = slate_items.iloc[order_indices].head(num_items_to_recom)
        recom_slate_indices = order_indices[:num_items_to_recom]
        slate_recom_indices = pd.DataFrame(
            {'slate_id': [slate_id] * num_items_to_recom, 'item_index': recom_slate_indices})
        recommendations_indices = pd.concat([recommendations_indices, slate_recom_indices])
        recommendations_features = pd.concat([recommendations_features, recom_slate_items_features])
        # print(recommendations))
    return recommendations_features, recommendations_indices

def train_pairwise_ranker_models(X_train, X_test, y_train, y_test, logistic_reg_model, user_types = ['Rational', 'Compromise', 'Similarity', 'Attraction']):

    results = dict((user_type, {'accuracy': 0, 'precision': 0}) for user_type in user_types)

    _X_train = X_train[['x_0', 'x_1', 'x_2', 'x_3', 'x_4']]
    _X_test = X_test[['x_0', 'x_1', 'x_2', 'x_3', 'x_4']]

    _X_test = _X_test.to_numpy()
    _X_train = _X_train.to_numpy()

    models = {}
    for user_type in user_types:
        # _y_train = y_train[user_type].to_numpy()
        _y_test = y_test[user_type].to_numpy()
        _y_train = y_train[[user_type, "slate_id"]].to_numpy()

        clf = logistic_reg_model.fit(_X_train, _y_train)
        models[user_type] = clf
        y_pred = clf.predict(_X_test)
        # print(f'y_pred = {y_pred}')
        precision = precision_score(_y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(_y_test, y_pred)
        results[user_type]['accuracy'] = accuracy
        results[user_type]['precision'] = precision

    return models, results