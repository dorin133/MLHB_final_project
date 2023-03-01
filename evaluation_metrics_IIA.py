import UserModel
import mnl
from ContextChoiceEnv import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score
import pairwise_ranker


# def kappa_eval_IIA(recommendations_features, recommendations_indices, num_rec_items, user_model1, user_model2):
#     count_IIA = 0
#     count_rep = 0
#     slate_id_arr = np.unique(recommendations_features.slate_id)
#     for curr_slate_id in slate_id_arr:
#         df_filter = recommendations_features[recommendations_features['slate_id'] == curr_slate_id]
#         df_filter_arr = df_filter.drop(columns=['slate_id']).to_numpy()
#         old_chosen_item = user_model.choice(df_filter_arr)
#         for item_idx in range(num_rec_items):
#             if old_chosen_item == item_idx:
#                 continue
#             count_rep += 1
#             df_filter_new = df_filter.drop(index = [df_filter.index[item_idx]], columns = ['slate_id']).to_numpy()
#             new_chosen_item = user_model.choice(df_filter_new)
#             if (df_filter_arr[old_chosen_item] == df_filter_new[new_chosen_item]).all():
#                 count_IIA += 1
#                 # print("YES")
#             # else:
#             #     print("NO")
#     return count_IIA / count_rep

def kappa_eval_IIA_2_models(recommendations_features_mnl, recommendations_features_ranker,
                            recommendations_indices_mnl, recommendations_indices_ranker,
                            num_rec_items, user_model):
    count_IIA_mnl = 0
    count_IIA_ranker = 0
    count_rep_ranker = 0
    count_rep_mnl = 0
    count_similar_recommendations = 0
    count_rep_similar = 0
    slate_id_arr = np.unique(recommendations_features_mnl.slate_id)
    for curr_slate_id in slate_id_arr:
        df_filter_mnl = recommendations_features_mnl[recommendations_features_mnl['slate_id'] == curr_slate_id]
        df_filter_ranker = recommendations_features_ranker[recommendations_features_ranker['slate_id'] == curr_slate_id]
        df_filter_arr_mnl = df_filter_mnl.drop(columns=['slate_id']).to_numpy()
        df_filter_arr_ranker = df_filter_ranker.drop(columns=['slate_id']).to_numpy()
        old_chosen_item_mnl = user_model.choice(df_filter_arr_mnl)
        old_chosen_item_ranker = user_model.choice(df_filter_arr_ranker)
        # if (df_filter_arr_mnl == df_filter_arr_ranker).all():
        #     count_similar_recommendations += 1
        if old_chosen_item_mnl == old_chosen_item_ranker:
            count_similar_recommendations += 1
        count_rep_similar += 1
        for item_idx in range(num_rec_items):
            if old_chosen_item_mnl == item_idx:
                continue
            count_rep_mnl += 1
            df_filter_new = df_filter_mnl.drop(index = [df_filter_mnl.index[item_idx]], columns = ['slate_id']).to_numpy()
            new_chosen_item = user_model.choice(df_filter_new)
            if (df_filter_arr_mnl[old_chosen_item_mnl] == df_filter_new[new_chosen_item]).all():
                count_IIA_mnl += 1

        for item_idx in range(num_rec_items):
            if old_chosen_item_ranker == item_idx:
                continue
            count_rep_ranker += 1
            df_filter_new = df_filter_ranker.drop(index = [df_filter_ranker.index[item_idx]], columns = ['slate_id']).to_numpy()
            new_chosen_item = user_model.choice(df_filter_new)
            if (df_filter_arr_ranker[old_chosen_item_ranker] == df_filter_new[new_chosen_item]).all():
                count_IIA_ranker += 1

    return count_IIA_mnl / count_rep_mnl, count_IIA_ranker / count_rep_ranker, count_similar_recommendations / count_rep_similar

# def main_real():
#     # user_models = {
#     #     'Attraction': AttractionUserModel(np.arange(num_features), 100),
#     #     'Compromise': CompromiseUserModel(np.arange(num_features), 100),
#     #     'Similarity': SimilarityUserModel(np.arange(num_features), 100),
#     #     'Rational': RationalUserModel(np.arange(num_features)),
#     # }
#     learning_models = {'logistic_reg': LogisticRegression(), 'pairwise_ranker': pairwise_ranker.MyRanker()}
#     env = TrainContextChoiceEnvironment()
#     # generate data
#     X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=7)
#
#     for learning_model_name in learning_models.keys():
#
#         if learning_model_name == "logistic_reg":
#             models, _ = mnl.train_mnl_models(X_train, X_test, y_train, y_test, learning_models[learning_model_name])
#         else:
#             models, _ = pairwise_ranker.train_pairwise_ranker_models(X_train, X_test, y_train, y_test,
#                                                                      learning_models[learning_model_name])
#
#         num_items_to_recom = 6
#
#         for user_model_name in env.user_models.keys():
#             if learning_model_name == "logistic_reg":
#                 recommendations_features, recommendations_indices = mnl.recommend_items_mnl(models[user_model_name], X_test, num_items_to_recom)
#             else:
#                 recommendations_features, recommendations_indices = pairwise_ranker.recommend_items_ranker(models[user_model_name], X_test, num_items_to_recom)
#
#             kappa_value_mnl, kappa_value_ranker = kappa_eval_IIA(recommendations_features = recommendations_features,
#                                         recommendations_indices = recommendations_indices,
#                                         num_rec_items = num_items_to_recom, user_model=env.user_models[user_model_name])
#             print(str(user_model_name) + ": kappa-mnl - " + str(kappa_value_mnl) + " kappa-ranker - " + str(kappa_value_ranker))
#             # print("similar ratio: "+ str(similar_ratio))
#             print("---------------")

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
    X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=20)

    models_mnl, _ = mnl.train_mnl_models(X_train, X_test, y_train, y_test, LogisticRegression())
    models_ranker, _ = pairwise_ranker.train_pairwise_ranker_models(X_train, X_test, y_train, y_test, pairwise_ranker.MyRanker())

    num_items_to_recom = 5

    for user_model_name in env.user_models.keys():
        recommendations_features_mnl, recommendations_indices_mnl = mnl.recommend_items_mnl(models_mnl[user_model_name], X_test, num_items_to_recom)
        recommendations_features_ranker, recommendations_indices_ranker = pairwise_ranker.recommend_items_ranker(models_ranker[user_model_name], X_test, num_items_to_recom)

        kappa_value_mnl, kappa_value_ranker, similar_ratio = kappa_eval_IIA_2_models(recommendations_features_mnl = recommendations_features_mnl,
                                                recommendations_features_ranker = recommendations_features_ranker,
                                                recommendations_indices_mnl = recommendations_indices_mnl,
                                                recommendations_indices_ranker =recommendations_indices_ranker,
                                                num_rec_items = num_items_to_recom, user_model=env.user_models[user_model_name])
        # kappa_value_ranker = kappa_eval_IIA(recommendations_features=recommendations_features_ranker,
        #                              recommendations_indices=recommendations_indices_ranker,
        #                              num_rec_items=num_items_to_recom, user_model=env.user_models[user_model_name])

        print(str(user_model_name) + ": kappa-mnl - " + str(kappa_value_mnl) + " kappa-ranker - " + str(
            kappa_value_ranker))
        print("similar ratio: " + str(similar_ratio))
        print("---------------")

if __name__ == "__main__":
    main()