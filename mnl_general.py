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
        
    @abc.abstractclassmethod
    def fit(self, X, y = None, varnames = ['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE'], choice_col = 'Rational'):
        pass
        
    def predict_proba(self, X, varnames = ['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT', 'HE']):
        choices, proba = self.model.predict(X[varnames], varnames, alts = X['alt'], ids = X['sample_id'], return_proba = True)
        return choices, proba
    
    def kappa(self, X, choices, user_model = 'Rational'):
        user_model = self.user_models[user_model]
        grouped_X = X.groupby('sample_id')
        chosen, _ = self.predict_proba(X)
        for samlpe_id, sample in grouped_X:
            old_chosen = chosen[samlpe_id]
            
         
        