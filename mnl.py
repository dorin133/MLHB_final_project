from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score

import numpy as np 
import pandas as pd


def train_mnl_models(X_train, X_test, y_train, y_test, mnl_model, user_types = ['Rational','Compromise', 'Similarity', 'Attraction']):

  results = dict((user_type,{'accuracy': 0, 'precision': 0}) for user_type in user_types)

  _X_train = X_train[['x_0', 'x_1', 'x_2', 'x_3','x_4']]
  _X_test =  X_test[['x_0', 'x_1', 'x_2', 'x_3','x_4']]

  models = {}
  for user_type in user_types:
    _y_train = y_train[user_type].to_numpy()
    _y_test = y_test[user_type].to_numpy()

    clf = mnl_model.fit(_X_train, _y_train)
    models[user_type] = clf
    y_pred = clf.predict(_X_test)
    # print(f'y_pred = {y_pred}')
    precision = precision_score(_y_test,y_pred, zero_division=0)
    accuracy  = accuracy_score(_y_test,y_pred)
    results[user_type]['accuracy'] = accuracy
    results[user_type]['precision'] = precision

  return models, results

def recommend_items_mnl(model, X_test, num_items_to_recom = 3):
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