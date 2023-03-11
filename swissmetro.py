import pandas as pd
import numpy as np

from xlogit.utils import wide_to_long

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

from UserModel import *

class SwissMetro():
    def __init__(self, num_features = 5, varnames = ['ASC_CAR', 'ASC_TRAIN', 'ASC_SM', 'R', 'CO', 'TT', 'HE']):
        num_features = len(varnames)
        beta_h = np.concatenate(([0.5, 0.5, 0.5], np.linspace(1, 3, num_features-3)))
        self.user_models = {
            'Rational' : RationalUserModel(beta_h), 
            'Compromise' : CompromiseUserModel(beta_h, 100),
            'Similarity' : SimilarityUserModel(beta_h, 100),
            'Attraction' : AttractionUserModel(beta_h,100),
            }
        self.varnames = varnames 
        self._prepare_data()
        
    def _train_test_split(self, df): # used to split the swissmetro data to train - test
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=123)
            split_indices = list(gss.split(df, groups=df['slate_id']))

            train_split_indices, test_split_indices  = split_indices[0]

            train_df, test_df = df.iloc[train_split_indices], df.iloc[test_split_indices]
            self.train_df = train_df
            self.test_df = test_df
            return train_df, test_df
    
    def _user_model_choices(self, df):
        alts = ['TRAIN', 'SM', 'CAR']
        grouped_df = df.groupby('sample_id')
        
        choices = {'Rational': [], 'Attraction': [], 'Compromise': [], 'Similarity': []}
        for id, group in grouped_df:
            for user_name, user_model in self.user_models.items():
                user_choice_idx = user_model.choice(group[self.varnames].values)
                user_choices = [alts[user_choice_idx]]*3 
                choices[user_name].extend(user_choices)
        choices_df = pd.DataFrame(choices)
        df = pd.concat([df, choices_df], axis =1)
        return df

    def _prepare_data(self):
        
        df_wide = pd.read_table("data/swissmetro.dat", sep='\t')

        # Keep only observations for commute and business purposes that contain known choices
        df_wide = df_wide[(df_wide['PURPOSE'].isin([1, 3]) & (df_wide['CHOICE'] != 0))]

        df_wide['sample_id'] = np.arange(len(df_wide))  # Add unique identifier
        df_wide['CHOICE'] = df_wide['CHOICE'].map({1: 'TRAIN', 2:'SM', 3: 'CAR'})
        df_wide.rename(columns = {'ID': 'slate_id'}, inplace = True)
        df = wide_to_long(df_wide, id_col='sample_id', alt_name='alt', sep='_',
            alt_list=['TRAIN', 'SM', 'CAR'], empty_val=0,
            varying=['TT', 'CO', 'HE', 'AV', 'SEATS'], alt_is_prefix=True)
        df['ASC_TRAIN'] = np.ones(len(df))*(df['alt'] == 'TRAIN')
        df['ASC_CAR'] = np.ones(len(df))*(df['alt'] == 'CAR')
        df['ASC_SM'] = np.ones(len(df))*(df['alt'] == 'SM')
        df['TT'], df['CO'], df['HE'], df['WHO'] = df['TT']/100, df['CO']/100, df['HE']/100, df['WHO']/3  # Scale variables
        # annual_pass = (df['GA'] == 1) & (df['alt'].isin(['TRAIN', 'SM']))
        # df.loc[annual_pass, 'CO'] = 0  # Cost zero for pass holders]
        # add a new column with random normal values
        df['R'] = np.random.normal(0, 1, len(df))
        df = self._user_model_choices(df)

        return self._train_test_split(df)

    def generate_datasets(self):
        X_train = self.train_df[['slate_id', 'AV', 'alt'] + self.varnames]
        X_test = self.test_df[['slate_id', 'AV', 'alt'] + self.varnames]
        y_train = self.train_df[['slate_id'] + list(self.user_models.keys())]
        y_test = self.test_df[['slate_id'] + list(self.user_models.keys())]
        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    sm_data = SwissMetro()
    X_train, X_test, y_train, y_test = sm_data.generate_datasets()
    print('..')
    