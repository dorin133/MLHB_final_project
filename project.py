#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# # Background and Research Question

# ## IIA

# Independence of Irrelevant Alternatives (IIA) is a principle in decision theory that states that the preference between two options should not be affected by the presence or absence of a third option that is not relevant to the decision at hand.
# 
# In the context of recommendation systems, IIA means that the ranking of items should not change based on the addition or removal of items that are not relevant to the user's preferences. In other words, if a user is given a set of items to choose from and they express a preference for one item over another, the introduction of additional items that are not relevant to their preferences should not change the relative ranking of the original items.
# 
# Simple predictive models do not account for context effects, such as the dependence of choice behavior on the set of alternatives in a choice set. However, real world user choises are usually set dependent; violating IIA.
# 
# In this study, we aim to study the relation between the learned model and the obtained user choices in its domain. Specifically, we will compare user choices under IIA vs. non-IIA prediction models, and try to observe if the learned model "cause" user choices to be more IIA, or vice versa.
# 
# 

# ## Research Question

# <center>Do users make more IIA choices when using an IIA model compared to non-IIA model? If so, are user choices really set independent or is it a delusion caused by the IIA model?<center>

# ## Hypothesis

# - We expect to observe IIA user choices in IIA domains    
# - More set dependent user choices in non IIA domains    
# - We expect that IIA user behavior is a delusion made by the IIA model recommendations – e.g. extreme score for chosen items?

# # Environment setup

# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import abc
# import itertools
# from sklearn import svm, linear_model
# import pandas as pd
# import sys

from ContextChoiceEnv import *
from UserModel import *


from collections import OrderedDict

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


env = ContextChoiceEnvironment()
env.inspect_data("current")


# # Experiments

# ## MNL model

# <center> <img src="imgs/MNL.jpg" /><center>

# ### MNL training

# In[ ]:


from mnl import train_logistic_regression_models
env = TrainContextChoiceEnvironment()
# generate data
X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=7)
# print(X_train)
# print(y_train)
models, mnl_fit_results = train_logistic_regression_models(X_train, X_test, y_train, y_test)
print(mnl_fit_results)


# In[ ]:


from mnl import recommend_items
# print(X_test)
recommendations_features, recommendations_indices = recommend_items(models['Rational'], X_test)
print(recommendations_features)
print(recommendations_indices)


# ### MNL domain - IIA evaluation

# ## Mixed MNL Model
# 

# # Discussion
