import UserModel
from trained_models_utils import *

from ContextChoiceEnv import *
from swissmetro import SwissMetro

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

import pairwise_ranker

from mnl_xlogit import MNL
from mixed_mnl_xlogit import Mixed_MNL

def recommendations_statistics(recommendations_features_all_models,  user_models = ['Compromise', 'Similarity']):
    total_stat = {}
    for user in user_models:
        user_stat = pd.DataFrame(columns=['slate_id', 'score'],)
        mnl_recom = recommendations_features_all_models['mnl'][user]
        mixed_mnl_recom = recommendations_features_all_models['mixed_mnl'][user]
        
        grouped_mnl_recom, grouped_mixed_mnl_recom = mnl_recom.groupby('slate_id'), mixed_mnl_recom.groupby('slate_id')    
        for slate_id in mnl_recom['slate_id'].unique():
            mnl_slate_recom = grouped_mnl_recom.get_group(slate_id)
            mixed_mnl_slate_recom = grouped_mixed_mnl_recom.get_group(slate_id)
            
            shared_recoms = pd.merge(mnl_slate_recom, mixed_mnl_slate_recom, how='inner')
            score = len(shared_recoms) / len(mnl_slate_recom)
            
            user_stat = pd.concat([user_stat, pd.DataFrame({'slate_id': [slate_id], 'score': [score]})])
        total_stat[user] = user_stat
    return total_stat 
            
def delusion_score(recommendations_features, num_rec_items, user_model, rational_user, varnames = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4' ]):
    count_IIA = 0
    dists_from_chosen_item = []
    slate_id_arr = np.unique(recommendations_features.slate_id)
    delusion_score_min = 0
    delusion_score_max = 0
    delusion_score_avg = 0
    for curr_slate_id in slate_id_arr:
        df_filter = recommendations_features[recommendations_features['slate_id'] == curr_slate_id]
        df_filter_arr = df_filter[varnames].to_numpy()
        old_chosen_item = user_model.choice(df_filter_arr)
        user_coice_coef_vanilla = rational_user.choice_coef(df_filter_arr)
        for item_idx in range(num_rec_items):
            if old_chosen_item == item_idx:
                continue
            dists_from_chosen_item.append(np.abs(user_coice_coef_vanilla[old_chosen_item] - user_coice_coef_vanilla[item_idx]))
        # print("the coef of user's choice: " + str(user_coice_coef_vanilla[old_chosen_item]) + " the coefs of items: " + str(user_coice_coef_vanilla))
        # print(dists_from_chosen_item)
        delusion_score_min += np.abs(np.min(np.array(dists_from_chosen_item)))
        delusion_score_max += np.abs(np.max(np.array(dists_from_chosen_item)))
        delusion_score_avg += np.abs(np.mean(np.array(dists_from_chosen_item)))
        dists_from_chosen_item = []
        # print("################")
    return delusion_score_min / slate_id_arr.shape[0], delusion_score_max / slate_id_arr.shape[0], delusion_score_avg / slate_id_arr.shape[0]


def kappa_eval_IIA_(recommendations_features, num_rec_items, user_model, varnames = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4' ]):
    count_IIA = 0
    count_rep = 0
    slate_id_arr = np.unique(recommendations_features.slate_id)
    for curr_slate_id in slate_id_arr:
        df_filter = recommendations_features[recommendations_features['slate_id'] == curr_slate_id]
        old_chosen_item = user_model.choice(df_filter[varnames].to_numpy())
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


def kappa_eval_IIA(recommendations_features, num_rec_items, user_model, varnames = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4' ]):
    grouped_recommendations = recommendations_features.groupby('slate_id')
    
    IIA_counter = 0 
    c = 0 
    for _, slate_recom in grouped_recommendations:
        original_slate_recom = slate_recom[varnames]
        chosen_idx = user_model.choice(original_slate_recom.to_numpy())
        original_chosen_idx = original_slate_recom.index[chosen_idx]
        for idx in range(num_rec_items):
            if idx == chosen_idx:
                continue
            c+=1
            new_slate_recom = original_slate_recom.drop(index = original_slate_recom.index[idx])
            new_chosen_idx = user_model.choice(new_slate_recom.to_numpy())
            if (new_slate_recom.index[new_chosen_idx] == original_chosen_idx):
                IIA_counter+=1
    return IIA_counter/c


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

    # learning_models = {'logistic_reg': LogisticRegression(), 'pairwise_ranker': pairwise_ranker.MyRanker(),
    #                    'svm_linear': OneVsRestClassifier(SVC(kernel='linear', probability=True)), 'svm_rbf': OneVsRestClassifier(SVC(kernel='rbf', gamma=0.5, C=0.5, probability=True))}
    data = 'swissmetro'
    
    # learning_models = {'mnl': MNL(), 'mixed_mnl': Mixed_MNL() }
    learning_models = { 'mnl': MNL() , 'mixed_mnl': Mixed_MNL()}
    
    if data == 'swissmetro':
        varnames = ['slate_id', 'AV', 'alt', 'ASC_CAR', 'ASC_TRAIN', 'ASC_SM', 'R', 'CO', 'TT', 'HE']
        varnames_ = ['ASC_CAR', 'ASC_TRAIN', 'ASC_SM', 'R', 'CO', 'TT', 'HE']
        env = SwissMetro(varnames = varnames_)
        X_train, X_test, y_train, y_test = env.generate_datasets()
        # X_train, X_test, y_train, y_test =  X_train[:54], X_test[:54], y_train[:54], y_test[:54]
    else:
        varnames = ['x_0', 'x_1', 'x_2', 'x_3','x_4']
        varnames_ = ['x_0', 'x_1', 'x_2', 'x_3','x_4']
        env = TrainContextChoiceEnvironment()
        # # generate data
        X_train, X_test, y_train, y_test = env.generate_datasets(num_features=5, num_items=20)

    num_items_to_recom = 10

    recommendations_features_all_models = {}
    for model_name, model_init in learning_models.items():
        recommendations_features_all_models[model_name] = recommend_items_wraper(X_train, X_test, y_train, y_test, model_init, num_items_to_recom=num_items_to_recom, varnames = varnames)

    stats = recommendations_statistics(recommendations_features_all_models)
    print(stats)
    
    for user_model_name in env.user_models.keys():
        if user_model_name in  ['Rational', 'Attraction']:
            continue
        print("****************", user_model_name)
        for model_name in list(learning_models.keys()):
            
           # kappa_value_mnl, kappa_value_ranker, similar_ratio = kappa_eval_IIA_2_models(recommendations_features_model1 = recommendations_features_all_models,
           #                                          recommendations_features_model1 = recommendations_features_ranker,
           #                                          num_rec_items = num_items_to_recom, user_model=env.user_models[user_model_name])
            kappa_value_curr = kappa_eval_IIA(recommendations_features=recommendations_features_all_models[model_name][user_model_name],
                                         num_rec_items=num_items_to_recom, user_model=env.user_models[user_model_name], varnames = varnames_ )
            
            delusion_score_curr_min, delusion_score_curr_max, delusion_score_curr_avg = delusion_score(recommendations_features=recommendations_features_all_models[model_name][user_model_name],
                                         num_rec_items=num_items_to_recom, user_model=env.user_models[user_model_name], varnames=varnames_, rational_user = env.user_models["Rational"])

                                                                                                       
            print("model: " + str(model_name) + ", user_model: " + str(user_model_name) + ", kappa-value: " + str(kappa_value_curr))
            print("model: " + str(model_name) + ", user_model: " + str(user_model_name) + ", the min delusion-score: " + str(delusion_score_curr_min))
            print("model: " + str(model_name) + ", user_model: " + str(user_model_name) + ", the max delusion-score: " + str(delusion_score_curr_max))
            print("model: " + str(model_name) + ", user_model: " + str(user_model_name) + ", the avg delusion-score: " + str(delusion_score_curr_avg))
            print("---------------")

if __name__ == "__main__":
    main()