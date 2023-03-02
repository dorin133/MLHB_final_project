import UserModel
from trained_models_utils import *
from ContextChoiceEnv import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score
import pairwise_ranker
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier


def kappa_eval_IIA(recommendations_features, num_rec_items, user_model):
    count_IIA = 0
    count_rep = 0
    slate_id_arr = np.unique(recommendations_features.slate_id)
    for curr_slate_id in slate_id_arr:
        df_filter = recommendations_features[recommendations_features['slate_id'] == curr_slate_id]
        df_filter_arr = df_filter.drop(columns=['slate_id']).to_numpy()
        old_chosen_item = user_model.choice(df_filter_arr)
        for item_idx in range(num_rec_items):
            if old_chosen_item == item_idx:
                continue
            count_rep += 1
            df_filter_new = df_filter.drop(index = [df_filter.index[item_idx]], columns = ['slate_id']).to_numpy()
            new_chosen_item = user_model.choice(df_filter_new)
            if (df_filter_arr[old_chosen_item] == df_filter_new[new_chosen_item]).all():
                count_IIA += 1
                # print("YES")
            # else:
            #     print("NO")
    return count_IIA / count_rep

def kappa_eval_IIA_2_models(recommendations_features_model1, recommendations_features_model2,
                            num_rec_items, user_model):
    count_IIA_model1 = 0
    count_IIA_model2 = 0
    count_rep_model1 = 0
    count_rep_model2 = 0
    count_similar_recommendations = 0
    count_rep_similar = 0
    slate_id_arr = np.unique(recommendations_features_model1.slate_id)
    for curr_slate_id in slate_id_arr:
        df_filter_model1 = recommendations_features_model1[recommendations_features_model1['slate_id'] == curr_slate_id]
        df_filter_model2 = recommendations_features_model2[recommendations_features_model2['slate_id'] == curr_slate_id]
        df_filter_arr_model1 = df_filter_model1.drop(columns=['slate_id']).to_numpy()
        df_filter_arr_model2 = df_filter_model2.drop(columns=['slate_id']).to_numpy()
        old_chosen_item_model1 = user_model.choice(df_filter_arr_model1)
        old_chosen_item_model2 = user_model.choice(df_filter_arr_model2)
        # if (df_filter_arr_mnl == df_filter_arr_ranker).all():
        #     count_similar_recommendations += 1
        if old_chosen_item_model1 == old_chosen_item_model2:
            count_similar_recommendations += 1
        count_rep_similar += 1
        for item_idx in range(num_rec_items):
            if old_chosen_item_model1 == item_idx:
                continue
            count_rep_model1 += 1
            df_filter_new = df_filter_arr_model1.drop(index = [df_filter_arr_model1.index[item_idx]], columns = ['slate_id']).to_numpy()
            new_chosen_item = user_model.choice(df_filter_new)
            if (df_filter_arr_model1[old_chosen_item_model1] == df_filter_new[new_chosen_item]).all():
                count_IIA_model1 += 1

        for item_idx in range(num_rec_items):
            if old_chosen_item_model2 == item_idx:
                continue
            count_rep_model2 += 1
            df_filter_new = df_filter_model2.drop(index = [df_filter_model2.index[item_idx]], columns = ['slate_id']).to_numpy()
            new_chosen_item = user_model.choice(df_filter_new)
            if (df_filter_arr_model2[old_chosen_item_model2] == df_filter_new[new_chosen_item]).all():
                count_IIA_model2 += 1

    return count_IIA_model1 / count_rep_model1, count_IIA_model2 / count_rep_model2, count_similar_recommendations / count_rep_similar

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

    learning_models = {'logistic_reg': LogisticRegression(), 'pairwise_ranker': pairwise_ranker.MyRanker(),
                       'svm_linear': OneVsRestClassifier(SVC(kernel='linear', probability=True)), 'svm_rbf': OneVsRestClassifier(SVC(kernel='rbf', gamma=0.5, C=0.5, probability=True))}
    env = TrainContextChoiceEnvironment()
    # generate data
    X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=20)

    recommendations_features_all_models = {}
    models = []
    for model_name in list(learning_models.keys()):
        models.append({'model_name': model_name, 'model_init': learning_models[model_name]})
        recommendations_features_all_models[model_name] = recommend_items_wraper(X_train, X_test, y_train, y_test, models[-1], num_items_to_recom=5)

    num_items_to_recom = 5

    for model_name in list(learning_models.keys()):
        for user_model_name in env.user_models.keys():
           # kappa_value_mnl, kappa_value_ranker, similar_ratio = kappa_eval_IIA_2_models(recommendations_features_model1 = recommendations_features_all_models,
           #                                          recommendations_features_model1 = recommendations_features_ranker,
           #                                          num_rec_items = num_items_to_recom, user_model=env.user_models[user_model_name])
            kappa_value_curr = kappa_eval_IIA(recommendations_features=recommendations_features_all_models[model_name][user_model_name],
                                         num_rec_items=num_items_to_recom, user_model=env.user_models[user_model_name])

            print("model: " + str(model_name) + ", user_model: " + str(user_model_name) + ", kappa-value: " + str(kappa_value_curr))
            print("---------------")

if __name__ == "__main__":
    main()