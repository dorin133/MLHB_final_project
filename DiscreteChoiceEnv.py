#@title Context Environment
import numpy as np
# from UserModel import *


Counter = 0

class DiscreteChoiceEnvironment:
    """
    Generic class for discrete-choice dataset generation
    """
    n_features = 8
    observations_per_user = 10
    items_per_slate = 1
    train_user_proportion = 0.6
    

    def _generate_user_item_attributes(self, n_users):
        """
        Generate latent parameters for users and items.

        Parameters
        ----------
        n_users : int

        Output
        ------
        users : ndarray of shape (n_users, n_features)
        items : ndarray of shape
                (n_users, observations_per_user, items_per_slate, n_features)
        """
        users = np.random.normal(
            size=(
                n_users,
                self.n_features,
            ),
        )
        items = np.random.normal(
            size=(
                n_users,
                self.observations_per_user,
                self.items_per_slate,
                self.n_features,
            ),
        )
        return users, items

    def _choice(self, users, items):
        """
        Discrete choice function
        
        Parameters
        ----------
        users : ndarray of shape (n_users, n_features)
        items : ndarray of shape
                (n_users, observations_per_user, items_per_slate, n_features)

        Output
        ------
        choice : Dict[str -> ndarray of shape(n_users, observations_per_user)]
        """
        raise NotImplementedError
    
    def _generate_choice_dataset(self, n_users):
        """
        Generate choice dataset, formatted as pandas dataframe.
        """
        users, items = self._generate_user_item_attributes(n_users)
        choice_dct = self._choice(users, items)
        rows = []
        for i in range(n_users):
            for j in range(self.observations_per_user):
                dct = {}
                dct['user_id'] = f'{i}'
                dct['slate_id'] = f'{i}_{j}'
                for k in range(self.items_per_slate):
                    for l in range(self.n_features):
                        dct[f'x_{k},{l}'] = items[i,j,k,l]
                for choice_type, choice_matrix in choice_dct.items():
                    dct[choice_type] = choice_matrix[i,j]
                rows.append(dct)
        df = pd.DataFrame(rows)
        return df
    
    def generate_datasets(self, n_users):
        n_train_users = int(n_users*self.train_user_proportion)
        n_test_users = n_users - n_train_users
        return (
            self._generate_choice_dataset(n_train_users),
            self._generate_choice_dataset(n_test_users),
        )
