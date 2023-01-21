
import numpy as np
import pandas as pd
from DiscreteChoiceEnv import *
from UserModel import *

class ContextChoiceEnvironment(DiscreteChoiceEnvironment):
    """
    Dataset generator for binary choice with decision noise
    """
    def __init__(self,
                 slate_number=3,
                 observations_per_user=50,
                 num_items_omega=15):
      
        # self.noise_scale = noise_scale
        user_model = AttractionUserModel(
        beta_h=np.array([1,9]),
          alpha_attr=3,
      )

        self.user_model = user_model
        self.slate_number = slate_number
        self.observations_per_user = observations_per_user
        self._generate_ex1_items(slate_number=slate_number,
                                 num_items_omega=num_items_omega)
        self.Counter = 0


    def _generate_ex1_items(self, slate_number=3, num_items_omega=15):
      if slate_number == 1:
        self.omega = np.array([[1,3,3,4,10,4,2],[1,7.5,5,4,5,6,1]]) 
        self.E_s = np.array([10, 20, 80, 40, 30, 80, 100])
        self.omega_tag = np.array([[4,2.5,4.5],[4.5,6,5]])
        self.E_s_tag = np.array([5, 8, 3])

      elif slate_number == 2:
        self.omega = np.array([[3,8,8.5,4,10,8,2, 1],[8,7,4,4,5,6,1, 8]])
        self.E_s = np.array([10, 90, 30, 50, 30, 40,100,90])

        self.omega_tag = np.array([[4,3,4.5,6],[8,6,5,6]])
        self.E_s_tag = np.array([5, 8, 3, 9])

      else:
        np.random.seed(24) 
        num_items_omega = 15
        self.omega, self.omega_tag = list(), list()
        self.E_s, self.E_s_tag = list(), list()

        for i in range(self.observations_per_user):
          self.omega.append(np.hstack([np.zeros((num_items_omega, 1)),
                                        np.ones((num_items_omega, 1))*i,
                                        np.random.randint(
                                            20,size=(num_items_omega,2)),
                                       np.random.randint(
                                           20,100,size=(num_items_omega,1))
                                        ]
                                       )
          )
          
          self.omega_tag.append(np.hstack([np.zeros((100, 1)),
                              np.ones((100, 1))*i,
                              np.random.randint(
                                  20,size=(100,2)),
                              np.random.randint(20,size=(100,1))
                              ]
                              )
          )

        self.omega = np.vstack(self.omega).T
        self.omega_tag = np.vstack(self.omega_tag).T
        self._items_to_frame(self.omega,self.E_s, data_name="Omega")
        self._items_to_frame(self.omega_tag, self.E_s_tag, ord('I'),
                                     data_name="Omega_tag")

      return

    def _items_to_frame(self, omega, E_s, char_start=ord('A'), data_name="Omega"):

        data = pd.DataFrame(omega.T).reset_index()
        if len(data) < self.observations_per_user:
          data['user_id'] = np.zeros(len(data))
          data['slate_id'] = np.zeros(len(data))
          data['rational_user_val'] = omega.T@\
                            self.user_model.rational_utility_weights
          data["E_s"] = E_s

        else:
          data['rational_user_val'] = data.loc[:,[2,3]].values@\
                  self.user_model.rational_utility_weights

          data.rename(columns={0:'user_id', 1:'slate_id',
                               2:'x_0', 3:'x_1', 4:'E_s'},
                      inplace=True)

        data['name'] = data['index'].apply(lambda x: chr(char_start+x))
        
        if data_name == "Omega":
          self.omega_df = data
        else:
          self.omega_tag_df = data

        return data
    
    def generate_slate_data_set(self, slate_id):

          data = self.omega_df.loc[self.omega_df.slate_id == slate_id].reset_index().drop(columns=["level_0"])

          self.top_5 = data.rational_user_val.argsort()[-5:][::-1]
          # data['top_5'] = data['index'].apply(lambda x: int(x) in self.top_5.values)
          data['perceived_val'] = np.nan
          feature_cols = [col for col in data.columns if col.startswith('x')]
          data.loc[self.top_5, ['perceived_val']] = self.user_model(
              data.loc[self.top_5, feature_cols].values)
          data = self._choice(data)
          
          return data


    def inspect_data(self, items_type="current", slate_id=-1):
      
      if items_type == "current":
        data =  self._items_to_frame(self.omega,self.E_s, data_name="Omega")
      elif items_type == "tag":
        data =  self._items_to_frame(self.omega_tag, self.E_s_tag, ord('A') + len(self.E_s),
                                     data_name="Omega_tag") 
        

      return data      


    def generate_datasets(self, slate_id=-1):
      self.Counter = self.Counter + 1
      data = pd.DataFrame(self.omega.T).reset_index()
      
      if len(data) < self.observations_per_user:
        data['user_id'] = np.zeros(len(data))
        data['slate_id'] = np.zeros(len(data))
        data['name'] = data['index'].apply(lambda x: chr(ord('A')+x))
        data['rational_user_val'] = self.omega.T@\
                                    self.user_model.rational_utility_weights
        self.top_5 = data.rational_user_val.argsort()[-5:][::-1]
        # print(self.top_5)
        data['top_5'] = data['index'].apply(lambda x: x.index in self.top_5)
        data['perceived_val'] = np.nan
        data.loc[self.top_5, ['perceived_val']] = self.user_model(data.loc[self.top_5, [0,1]].values)
        res = self._choice(data)    
      
      else:
        # if slate_id != -1:
        res = self.generate_slate_data_set(slate_id)

      return res
      
    def _choice(self, data, chosen='chosen'):
        dummies = pd.get_dummies(data.index)
        # for name, group in grouped:
        self.chosen = data['perceived_val'].astype(float).idxmax()
        data[chosen] = dummies.iloc[data['perceived_val'].astype(float).idxmax(), :]
        return data

    def add_item(self, index):
      item = self.omega_tag_df.loc[self.omega_tag_df["index"] == index].copy()
      new_index = len(self.omega_df)
      item.index = [new_index]
      item['index'] = item.index
      self.omega_df = pd.concat([self.omega_df, item])
      return self.omega_df.loc[
          self.omega_df.slate_id == item.slate_id.iloc[0]]

    def drop_item(self, index):
      # to_pop = max(self.omega_df.index)
      # df_T = self.omega_df.T
      # df_T.pop(to_pop)
      # self.omega_df = df_T.T
      # return self.omega_df
      item_slate = self.omega_df.loc[self.omega_df.index == index].slate_id.iloc[0]
      self.omega_df.drop(index=index, inplace=True)
      return self.omega_df.loc[
          self.omega_df.slate_id == item_slate]
    
    def top_5_idx(self, rational_user_val):
      return rational_user_val.argsort()[-5:][::-1]


# a_user = AttractionUserModel(
#     beta_h=np.array([1,9]),
#     alpha_attr=3,
# )

# env = ContextChoiceEnvironment(user_model=a_user)
# d = env.inspect_data("current")
# env.generate_datasets(0)



#@title Train Context Environment

class TrainContextChoiceEnvironment(DiscreteChoiceEnvironment):
    """
    Dataset generator for binary choice with decision noise
    """
    def __init__(self,
                 observations_per_user=500,
                 num_items=7,
                 num_features=5):

        self.user_models = {
            'Attraction' : AttractionUserModel( np.arange(num_features),100),
            'Compromise' : CompromiseUserModel(np.arange(num_features), 100),
            'Similarity' : SimilarityUserModel(np.arange(num_features), 100),
            'Rational' : RationalUserModel(np.arange(num_features)),
            }

        self.observations_per_user = observations_per_user
        self.items_per_slate = num_items
        self.n_features = num_features
        self._generate_user_item_attributes(1)

    def generate_datasets(self, num_features=5, num_items=7):

          # df_train, df_test = list(), list()
          # num_slates = self.omega_df.slate_id.nunique()
          # train_slates = int(0.8*num_slates)
          
          # for slate in range(train_slates):
          #   df_train.append(self.generate_slate_data_set(slate))

          # for slate in range(train_slates, num_slates):
          #   df_test.append(self.generate_slate_data_set(slate))
          # train_data = pd.concat(df_train).drop(columns=['name', 'index',])
          # test_data = pd.concat(df_test)

          # train_label = train_data['chosen']
          # test_label = test_data['chosen']
          self.items_per_slate = num_items
          self.num_features = num_features

          train_slates = int(0.8*self.observations_per_user)
          train_data, y_train = self._generate_choice_dataset(num_features=num_features,
                                                     num_slates=train_slates,
                                                      num_items=num_items)

          test_slates = self.observations_per_user - train_slates
          test_data, y_test = self._generate_choice_dataset(num_features=num_features,
                                                    num_slates=test_slates,
                                                    num_items=num_items)



          return train_data, test_data, y_train, y_test


    def _choice(self, data):
        dummies = pd.get_dummies(range(len(data)))
        chosen = dummies[data.astype(float).argmax()]
        return chosen

    def _generate_choice_dataset(self, num_features=5, num_slates=500, num_items=7):
        """
        Generate choice dataset, formatted as pandas dataframe.
        """
        _, items = self._generate_user_item_attributes(0)
        

        y_train = {name:[] for name,user in self.user_models.items()}
        train = list()
        for i in range(num_slates):
            train.append(np.hstack(
                      [np.ones((num_items, 1))*i, # slate_id
                      np.random.normal(size=(num_items,num_features)), # data
                                                    ]))
            for name, user in self.user_models.items():
              y_train[name].append(self._choice(user(train[-1][:,1:])))
        
        feature_names = [f'x_{i}' for i in range(num_features)]
        col_names = ['slate_id'] + feature_names
        train_df = pd.DataFrame(np.vstack(train),
                                columns=col_names)

        for name, choices in y_train.items():
          train_df[name] = np.hstack(choices)

        train_data = train_df[col_names]#.drop(columns=['slate_id'])
        y = train_df[list(y_train.keys()) + ['slate_id']]
        return train_data, y

    def mean_welfare(self, X):
      return (X@np.arange(self.num_features)).mean()




# enV2 = TrainContextChoiceEnvironment(num_features=5, num_items=7)
# enV2.generate_datasets(num_features=5, num_items=7)

# add slate id to label, see which entry is value before doing so