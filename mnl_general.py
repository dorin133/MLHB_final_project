from xlogit.utils import wide_to_long

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

from UserModel import *
from trained_models_utils import *
from ContextChoiceEnv import *

class MNL_General():
    def __init__(self, num_features = 5):
        self.user_models = {
            'Rational' : RationalUserModel(np.arange(num_features)-5), 
            'Compromise' : CompromiseUserModel(np.arange(num_features), 100),
            'Similarity' : SimilarityUserModel(np.arange(num_features), 100),
            'Attraction' : AttractionUserModel( np.arange(num_features),100),
            }

    
    def _train_test_split(self, df): # used to split the swissmetro data to train - test
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=123)
        split_indices = list(gss.split(df, groups=df['ID']))

        train_split_indices, test_split_indices  = split_indices[0]

        train_df, test_df = df.iloc[train_split_indices], df.iloc[test_split_indices]
        self.train_df = train_df
        self.test_df = test_df
        return train_df, test_df
    
    def _user_model_choices(self, df, varnames = ['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE']):
        alts = ['TRAIN', 'SM', 'CAR']
        grouped_df = df.groupby('slate_id')
        
        choices = {'Rational': [], 'Attraction': [], 'Compromise': [], 'Similarity': []}
        for id, group in grouped_df:
            for user_name, user_model in self.user_models.items():
                user_choice_idx = user_model.choice(group[varnames].values)
                user_choices = [alts[user_choice_idx]]*3 
                choices[user_name].extend(user_choices)
        choices_df = pd.DataFrame(choices)
        df = pd.concat([df, choices_df], axis =1)
        return df
    
    def _prepare_data(self, X, y=None):
        
        if X == 'swissmetro':
            df_wide = pd.read_table("data/swissmetro.dat", sep='\t')

            # Keep only observations for commute and business purposes that contain known choices
            # df_wide = df_wide[(df_wide['PURPOSE'].isin([1, 3]) & (df_wide['CHOICE'] != 0))]
            df_wide = df_wide[(df_wide['CHOICE'] != 0)]

            df_wide['slate_id'] = np.arange(len(df_wide))  # Add unique identifier
            df_wide['CHOICE'] = df_wide['CHOICE'].map({1: 'TRAIN', 2:'SM', 3: 'CAR'})
            df = wide_to_long(df_wide, id_col='slate_id', alt_name='alt', sep='_',
                  alt_list=['TRAIN', 'SM', 'CAR'], empty_val=0,
                  varying=['TT', 'CO', 'HE', 'AV', 'SEATS'], alt_is_prefix=True)
            df['ASC_TRAIN'] = np.ones(len(df))*(df['alt'] == 'TRAIN')
            df['ASC_CAR'] = np.ones(len(df))*(df['alt'] == 'CAR')
            df['TT'], df['CO'], df['HE'] = df['TT']/100, df['CO']/100, df['HE']/100  # Scale variables
            annual_pass = (df['GA'] == 1) & (df['alt'].isin(['TRAIN', 'SM']))
            df.loc[annual_pass, 'CO'] = 0  # Cost zero for pass holders
            
            df = self._user_model_choices(df)
            
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
    
    @abc.abstractclassmethod
    def fit(self, X, y = None, varnames = ['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE'], choice_col = 'Rational'):
        pass
        
    def predict_proba(self, X, varnames = ['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE']):
        choices, proba = self.model.predict(X[varnames], varnames, alts = X['alt'], ids = X['slate_id'], return_proba = True)
        return choices, proba
    
    def kappa(self, X, choices, user_model = 'Rational'):
        user_model = self.user_models[user_model]
        grouped_X = X.groupby('slate_id')
        chosen, _ = self.predict_proba(X)
        for samlpe_id, sample in grouped_X:
            old_chosen = chosen[samlpe_id]
            
         
        