from xlogit import MixedLogit
from xlogit.utils import wide_to_long

import UserModel
from trained_models_utils import *
from ContextChoiceEnv import *

class Mixed_MNL():
    def __init__(self):
        self.model = MixedLogit()
        
    def _prepare_data(self, X, y=None):
        
        if X == 'swissmetro':
            df_wide = pd.read_table("http://transp-or.epfl.ch/data/swissmetro.dat", sep='\t')

            # Keep only observations for commute and business purposes that contain known choices
            df_wide = df_wide[(df_wide['PURPOSE'].isin([1, 3]) & (df_wide['CHOICE'] != 0))]

            df_wide['custom_id'] = np.arange(len(df_wide))  # Add unique identifier
            df_wide['CHOICE'] = df_wide['CHOICE'].map({1: 'TRAIN', 2:'SM', 3: 'CAR'})
            df = wide_to_long(df_wide, id_col='custom_id', alt_name='alt', sep='_',
                  alt_list=['TRAIN', 'SM', 'CAR'], empty_val=0,
                  varying=['TT', 'CO', 'HE', 'AV', 'SEATS'], alt_is_prefix=True)
            df['ASC_TRAIN'] = np.ones(len(df))*(df['alt'] == 'TRAIN')
            df['ASC_CAR'] = np.ones(len(df))*(df['alt'] == 'CAR')
            df['TT'], df['CO'] = df['TT']/100, df['CO']/100  # Scale variables
            annual_pass = (df['GA'] == 1) & (df['alt'].isin(['TRAIN', 'SM']))
            df.loc[annual_pass, 'CO'] = 0  # Cost zero for pass holders
            return df
        
        else: # generated data     
            num_slates =  X['slate_id'].nunique()
            num_alts =  X.shape[0]/ num_slates
            alts = np.tile(np.arange(1,num_alts+1), num_slates)
            choice = []
            for i in range(0, int(X.shape[0]), int(num_alts)):
                slate_choice = y[i:int((i+num_alts))]
                slate_choice_ind = np.argmax(slate_choice)
                alt_y = [slate_choice_ind]*int(num_alts)
                choice.extend(alt_y)
            return y, alts
    
    
    def fit(self, X, y = None, varnames = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4']):
        data = self._prepare_data(X, y)
        if X == 'swissmetro':
            self.model.fit(X=data[varnames], y=data['CHOICE'], varnames=varnames,
            alts=data['alt'], ids=data['custom_id'], avail=data['AV'],
            panels=data["ID"], randvars={'TT': 'n', 'HE': 'n'}, n_draws=500,
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
        
    def predict_proba(self, X, varnames = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4']):
        # num_slates =  X['slate_id'].nunique()
        # num_alts =  X.shape[0]/ num_slates
        # alts = np.tile(np.arange(num_alts), num_slates)
        # # alts = np.arange(len(X))
        # self.model.predict(X[varnames], varnames=varnames, alts = X.index, ids=X['slate_id'], return_proba=True)
        # coeffs = self.model.coeff_

        coeffs = pd.DataFrame(self.model.coeff_.reshape(1, -1), columns=self.model.coeff_names)
        mean_probas = None 
        for r in range(100):
            draws_weights = []
            for var in varnames : 
                mean = coeffs[var] 
                std = abs(coeffs['sd.' + var])
                var_weight = np.random.normal(loc = mean, scale=std)
                draws_weights.append(var_weight)
            scores = X[varnames]@draws_weights 
            if mean_probas is None:    
                mean_probas = scores
            else: 
                mean_probas =( mean_probas + scores)/(r+1)
        return mean_probas
        
if __name__ == '__main__':
    env = TrainContextChoiceEnvironment()
    # generate data
    # X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=20)
    model = Mixed_MNL()
    # fitted = model.fit(X_train, y_train['Rational'])
    fitted = model.fit('swissmetro', varnames=['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE'])
    # probas = model.predict_proba(X_test)

