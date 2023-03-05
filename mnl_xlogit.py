from xlogit import MultinomialLogit
from xlogit.utils import wide_to_long

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

import UserModel
from trained_models_utils import *
from ContextChoiceEnv import *

class MNL():
    def __init__(self):
        self.model = MultinomialLogit()
    
    def _train_test_split(self, df): # used to split the swissmetro data to train - test
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        split_indices = list(gss.split(df, groups=df['custom_id']))
        train_split_indices, test_split_indices  = split_indices[0]

        train_df, test_df = df.iloc[train_split_indices], df.iloc[test_split_indices]
        self.train_df = train_df
        self.test_df = test_df
        return train_df, test_df

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
            
            train_df, test_df = self._train_test_split(df)
            return train_df, test_df
        
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
    
    
    def fit(self, X, y = None, varnames = ['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE']):
        data = self._prepare_data(X, y)
        if X == 'swissmetro':
            train_df, _ =  data
            self.model.fit(X=train_df[varnames], y=train_df['CHOICE'], varnames=varnames,
            alts=train_df['alt'], ids=train_df['custom_id'], avail=train_df['AV'])
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
        
    def predict_proba(self, X, varnames = ['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE']):
        _, proba = self.model.predict(X[varnames], varnames, alts = X['alt'], ids = X['custom_id'], return_proba = True)
        return proba
        
if __name__ == '__main__':
    # env = TrainContextChoiceEnvironment()
    # generate data
    # X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=20)
    model = MNL()
    # fitted = model.fit(X_train, y_train['Rational'])
    fitted = model.fit('swissmetro', varnames=['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE'])
    probas = model.predict_proba(model.test_df)

