from xlogit import MixedLogit
from mnl_general import MNL_General
import pandas as pd
import numpy as np
class Mixed_MNL(MNL_General):
    def __init__(self):
        super(). __init__()
        self.model = MixedLogit()
    
    def fit(self, X, y = None, varnames = ['ASC_CAR', 'ASC_TRAIN', 'ASC_SM', 'R', 'CO', 'TT', 'HE']):

        self.randvars = {'R': 'n', 'CO':'n', 'TT': 'n', 'HE': 'n'}
        self.model.fit(X=X[varnames], y=y, varnames=varnames,
        alts=X['alt'], ids=X['slate_id'], avail=X['AV'],
        randvars=self.randvars , n_draws=500, optim_method='L-BFGS-B')
        self.model.summary()
        return self
    
    def predict_proba(self, X, num_draws = 200, varnames = ['ASC_CAR', 'ASC_TRAIN', 'ASC_SM', 'R', 'CO', 'TT', 'HE']):
        coeffs = pd.DataFrame(self.model.coeff_.reshape(1, -1), columns=self.model.coeff_names)
        mean_scores = None 
        for r in range(num_draws):
            draws_weights = []
            for var in varnames : 
                mean = coeffs[var] 
                if var in self.randvars:
                    std = abs(coeffs['sd.' + var])
                    var_weight = np.random.normal(loc = mean, scale=std)
                else:
                    var_weight = coeffs[var]
                draws_weights.append(var_weight)
            scores = X[varnames]@draws_weights 
            if mean_scores is None:    
                mean_scores = scores
            else: 
                mean_scores =(mean_scores + scores)/(r+1)
                
        # probas = np.exp(mean_scores) / np.sum(np.exp(mean_scores))
        # comp_probas = 1 - probas
        return np.column_stack([mean_scores, mean_scores])        
if __name__ == '__main__':
    # env = TrainContextChoiceEnvironment()
    # generate data
    # X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=20)
    model = Mixed_MNL()
    # fitted = model.fit(X_train, y_train['Rational'])
    fitted = model.fit('swissmetro', varnames=['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE'])
    probas = model.predict_proba(model.test_df)
    print('Done')
