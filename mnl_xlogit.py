from xlogit import MultinomialLogit
import numpy as np
from mnl_general import MNL_General
class MNL(MNL_General):
    def __init__(self, num_features = 5):
        super(). __init__()
        self.model = MultinomialLogit()
    
    def fit(self, X, y = None, varnames = ['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE'], choice_col = 'Rational'):
        data = self._prepare_data(X, y)
        if X == 'swissmetro':
            train_df, _ =  data
            print(train_df.shape[0])
            print(train_df['ID'].value_counts())
            self.model.fit(X=train_df[varnames], y=train_df[choice_col], varnames=varnames,
            alts=train_df['alt'], ids=train_df['ID'], avail=train_df['AV'])
            self.model.summary()
        else: # generated data 
            choice, alts  = data
            randvars = {}
            for var in varnames:
                randvars[var] = 'n'
                
            self.model.fit(X=X[varnames], y=choice, 
            varnames=varnames,
            alts=alts, ids=X['slate_id'],
            randvars=randvars,
            n_draws=600,
            optim_method='L-BFGS-B'
            )
            self.model.summary()
            return self
        
        
def SB(choices):
    alts = ['CAR', 'SM', 'TRAIN']
    sb = 0 
    for i in alts:
        for j in alts:
            if i == j:
                continue
            X_ij = np.sum(choices == i) 
            N_ij = np.sum(choices == i) + np.sum(choices == j)
            if N_ij == 0 or X_ij == 0: 
                continue
            P_ij = X_ij / N_ij
            SB_ij = (P_ij*N_ij)
            SB_ij = (X_ij - P_ij*N_ij)
            SB_ij = (X_ij - P_ij*N_ij)**2
            SB_ij = (X_ij - P_ij*N_ij)**2 / (P_ij*N_ij)
            sb+=SB_ij
    return sb

if __name__ == '__main__':
    # env = TrainContextChoiceEnvironment()
    # generate data
    # X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=20)
    model = MNL()
    # fitted = model.fit(X_train, y_train['Rational'])
    fitted = model.fit('swissmetro', varnames=['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE'], choice_col = 'Attraction')
    choices, probas = model.predict_proba(model.test_df)
    print('..')

