from xlogit import MixedLogit
from mnl_general import MNL_General

class Mixed_MNL(MNL_General):
    def __init__(self):
        super(). __init__()
        self.model = MixedLogit()
    
    def fit(self, X, y = None, varnames = ['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE'], choice_col = 'Rational'):
        data = self._prepare_data(X, y)
        if X == 'swissmetro':
            train_df, _ =  data
            self.model.fit(X=train_df[varnames], y=train_df['CHOICE'], varnames=varnames,
            alts=train_df['alt'], ids=train_df['slate_id'], avail=train_df['AV'],
            panels=train_df["ID"], randvars={'TT': 'n', 'HE': 'n'}, n_draws=500,
            optim_method='L-BFGS-B')
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
        
        
        
if __name__ == '__main__':
    # env = TrainContextChoiceEnvironment()
    # generate data
    # X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=20)
    model = Mixed_MNL()
    # fitted = model.fit(X_train, y_train['Rational'])
    fitted = model.fit('swissmetro', varnames=['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE'])
    choices, probas = model.predict_proba(model.test_df)
    model.kappa(choices)

