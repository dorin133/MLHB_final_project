import UserModel
import mnl
from ContextChoiceEnv import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score
import pairwise_ranker


def kappa_eval_IIA(recommendations_features, recommendations_indices, num_rec_items, user_model):
    count_IIA = 0
    for curr_item_num, curr_slate_id in zip(list(recommendations_features.index), list(recommendations_features.slate_id)):
        df_filter = recommendations_features[recommendations_features['slate_id'] == curr_slate_id]
        old_chosen_item = user_model.choice(df_filter.drop(columns = ['slate_id']).to_numpy())
        new_chosen_item = user_model.choice(df_filter.drop(index = [curr_item_num], columns = ['slate_id']).to_numpy())
        if old_chosen_item == new_chosen_item:
            count_IIA += 1
            # print("YES")
        # else:
        #     print("NO")
    return count_IIA / (recommendations_indices.shape[0])


def main():
    # user_models = {
    #     'Attraction': AttractionUserModel(np.arange(num_features), 100),
    #     'Compromise': CompromiseUserModel(np.arange(num_features), 100),
    #     'Similarity': SimilarityUserModel(np.arange(num_features), 100),
    #     'Rational': RationalUserModel(np.arange(num_features)),
    # }
    learning_models = {'logistic_reg': LogisticRegression(), 'pairwise_ranker': pairwise_ranker.MyRanker()}
    env = TrainContextChoiceEnvironment()
    # generate data
    X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=7)
    _X_test = X_test[['x_0', 'x_1', 'x_2', 'x_3', 'x_4']]

    for learning_model_name in learning_models.keys():

        models, mnl_fit_results = mnl.train_logistic_regression_models(X_train, X_test, y_train, y_test, learning_models[learning_model_name], learning_model_name)
        num_items_to_recom = 3

        for user_model_name in env.user_models.keys():
            if learning_model_name == "logistic_reg":
                recommendations_features, recommendations_indices = mnl.recommend_items(models[user_model_name], X_test, num_items_to_recom)
            else:
                recommendations_features, recommendations_indices = models[user_model_name].recommend_items_ranker(X_test)
            # print(recommendations_features)
            # print(recommendations_indices)
            kappa_value = kappa_eval_IIA(recommendations_features = recommendations_features,
                                recommendations_indices = recommendations_indices,
                                num_rec_items = num_items_to_recom, user_model=env.user_models[user_model_name])
            print(str(learning_model_name)+' & '+str(user_model_name) + ": "+ str(kappa_value))
            print("---------------")

if __name__ == "__main__":
    main()