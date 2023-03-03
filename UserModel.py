
import abc
import numpy as np

class UserModel(abc.ABC):
  @abc.abstractclassmethod
  def __call__(self, X, *args):
    '''
    Given items X, calculate the user's valuation v(x) for each item x
    '''
    raise NotImplementedError()
  
  def predict(self, X, *args):
    raise NotImplementedError()

class RationalUserModel(UserModel):

  def __init__(self, beta_h):
    self.beta_h = beta_h
    self.type = "rational"

  def __call__(self, X):
    return X@self.beta_h
  
  def choice(self, X):
    return np.argmax(self(X), axis=0)

  def choice_coef(self, X):
    return self(X)


class AttractionUserModel(UserModel):
  def __init__(self, beta_h, alpha_attr):
    # rational_utility_weights
    self.rational_utility_weights = beta_h 
    # attraction_coefficient
    self.attraction_coefficient = alpha_attr
    self.name = "attraction"

  def __call__(self, X):
    # X: shape=(num_items,num_features) - Item covariates
    return (
        self._rational_decision_term(X)
        + self.attraction_coefficient*self._attraction_term(X)
    )

  def _rational_decision_term(self, X):
    return X@self.rational_utility_weights

  def _attraction_term(self, X):
    # Calculate preference vector
    preference_vector = X.max(axis=0)-X.min(axis=0)
    preference_vector_normalized = (
        preference_vector
        /np.sqrt(preference_vector@preference_vector)
    )
    # Calculate dominance and distance matrices
    N = len(X)
    dominance = np.zeros((N,N))
    distance = np.zeros((N,N))
    for i in range(N):
      for j in range(N):
        dominance[i,j] = (+1)*np.all(X[i]-X[j] >= 0) + (-1)*np.all(X[j]-X[i] >= 0)
        distance[i,j] = abs(preference_vector_normalized@(X[i]-X[j]))

    attraction_terms = (dominance*distance).sum(axis=1)
    return attraction_terms
  
  def choice(self, X):
    return np.argmax(self(X), axis=0)

  def choice_coef(self, X):
    return self(X)



class SimilarityUserModel(UserModel):

  def __init__(self, beta_h, beta_sim):
    self.beta_h = beta_h
    self.beta_sim = beta_sim
    self.name = "similarity"

  def __call__(self, X):

    # calculate actual value
    v_ih = X@self.beta_h

    # calculate perference vector
    pref_vec = X.max(axis=0) - X.min(axis=0)


    # project on the ortogonal hyperplane
    X_projected = np.zeros(X.shape)
    projection_size = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        X_projected[i, :] = X[i,:] -  (X[i,:].dot(pref_vec) / np.linalg.norm(pref_vec)) * pref_vec

    # find min distance on projected

    distances = np.zeros((X_projected.shape[0], X_projected.shape[0]))

    for i in range(X_projected.shape[0]): # could be optimized (cal half mat etc..)
      for j in range(X_projected.shape[0]):
        distances[i,j] = np.linalg.norm(X_projected[i,:] - X_projected[j,:])
    
    # avoide i == j
    distances = distances + np.eye(distances.shape[0])*distances.max()

    min_distances = distances.min(axis=1) # go over columns

    return v_ih + self.beta_sim*-min_distances
  
  def choice(self, X):
    return np.argmax(self(X))

  def choice_coef(self, X):
    return self(X)


class CompromiseUserModel(UserModel):

  def __init__(self, beta_h, beta_com):
    self.beta_h = beta_h
    self.beta_com = beta_com
    self.name = 'compromise'

  def __call__(self, X):

    # calculate rational value
    v_ih = X@self.beta_h
    # calculate compromise value
    X_com = (X.max(axis=0) + X.min(axis=0)) / 2

    # pairwise distance between rows of X and X_com
    d_im = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        d_im[i] = np.linalg.norm(X[i,:] - X_com)

    return v_ih + -d_im*self.beta_com
  
  def x_user_choice_positions(self, X):
    return np.argmax(self(X), axis=0)

  def choice(self, X):
    return np.argmax(self(X), axis=0)

  def choice_coef(self, X):
    return self(X)