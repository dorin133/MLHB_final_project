from xlogit import MultinomialLogit
import numpy as np
from mnl_general import MNL_General
from swissmetro import SwissMetro
class MNL(MNL_General):
    def __init__(self, num_features = 5):
        super(). __init__()
        self.model = MultinomialLogit()
    
    def fit(self, X, y = None, varnames = ['ASC_CAR', 'ASC_TRAIN', 'ASC_SM', 'R', 'CO', 'TT', 'HE']):
        # if X == 'swissmetro':
        self.model.fit(X=X[varnames], y=y, varnames=varnames,
        alts=X['alt'], ids=X['slate_id'], avail=X['AV'])
        self.model.summary()
        return self
    
    def predict_proba(self, X, varnames = ['ASC_CAR', 'ASC_TRAIN', 'ASC_SM', 'R', 'CO', 'TT', 'HE']):
        coeffs = self.model.coeff_
        scores = X[varnames] @ coeffs
        # probas = np.exp(scores) / np.sum(np.exp(scores))
        # comp_probas = 1 - probas
        # 1st com is probability of not being chosen
        # return np.column_stack([comp_probas, probas])
        return np.column_stack([scores, scores])

if __name__ == '__main__':
    env = SwissMetro(varnames)
    X_train, X_test, y_train, y_test = env.generate_datasets()
    model = MNL()
    # fitted = model.fit(X_train, y_train['Rational'])
    fitted = model.fit(X_train, y_train, varnames=['R', 'ASC_CAR', 'ASC_TRAIN', 'ASC_SM', 'CO', 'TT', 'HE'])
    probas_table = model.predict_proba(model.test_df[model.test_df['slate_id']==6])
    probas = probas_table[:, 0]
    print('Done')
