from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score

def train_logistic_regression_models(X_train, X_test, y_train, y_test):
  user_types = ['Rational','Compromise', 'Similarity', 'Attraction']
  results = dict((user_type,{'accuracy': 0, 'precision': 0}) for user_type in user_types)

  _X_train = X_train[['x_0', 'x_1', 'x_2', 'x_3','x_4']]
  _X_test =  X_test[['x_0', 'x_1', 'x_2', 'x_3','x_4']]

  for user_type in user_types:
    _y_train = y_train[user_type]
    _y_test = y_test[user_type]
    
    clf = LogisticRegression().fit(_X_train, _y_train)
    y_pred = clf.predict(_X_test)
    # print(f'y_pred = {y_pred}')
    precision = precision_score(_y_test,y_pred, zero_division=0) 
    accuracy  = accuracy_score(_y_test,y_pred) 
    results[user_type]['accuracy'] = accuracy  
    results[user_type]['precision'] = precision 

  return results