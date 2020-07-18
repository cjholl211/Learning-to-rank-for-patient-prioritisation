from __future__ import print_function
# skip all warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import tensorflow as tf
from DirectRanker import directRanker
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from functools import partial
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from dr_helpers import nDCG_cls, np_append, cmp_to_key, nDCG_, compare_net

# because we have no ground truth, we must loop through a variety of severity scores in place
score_names = ['oasis', 'saps', 'qsofa', 'sofa', 'los']

for score_name in score_names:

    # features for DirectRanker
    col_list = ['hadm_id', 'vent', 'heartrate_mean', 'sysbp_mean', 'spo2_mean', 'resprate_mean', 'mild_gcs', 'moderate_gcs', 'severe_gcs', 'tempc_mean']
    mimic_data = pd.read_csv('unscaled_ltr_data.csv', sep=',', usecols=col_list)

    # get NEW2 scores so we can use the same splits
    col_list = ['hadm_id', 'news2_score']
    news2 = pd.read_csv('news2_conversion_scores.csv', sep=',', usecols=col_list)

    # LoS is not stored with severity scores
    if score_name != 'los':

        col_list = ['hadm_id', score_name]
        severity_scores = pd.read_csv('severity_scores_hadm_id.csv', sep=',', usecols=col_list)

        severity_scores = severity_scores.groupby(['hadm_id']).mean().round()

        mimic_data_score = pd.merge(severity_scores, mimic_data, on='hadm_id', how='inner')

        # combine data set with ground truth to measure NDCG from NEWS2
        news2 = pd.merge(news2, severity_scores, how='inner', on='hadm_id')

        target = mimic_data_score[[score_name]]
        # scale targets before split
        lmm_scaler = MinMaxScaler()
        lmm_scaler.fit(target[[score_name]])
        target[score_name] = lmm_scaler.transform(target[[score_name]])
        news2[score_name] = lmm_scaler.transform(news2[[score_name]])

    else:

        col_list = ['hadm_id', score_name]
        los = pd.read_csv('death_within_3_days.csv', sep=',', usecols=col_list)
        los.groupby('hadm_id', axis=1).mean().round()

        mimic_data_score = pd.merge(los, mimic_data, on='hadm_id', how='inner')

        news2 = pd.merge(news2, los, how='inner', on='hadm_id')

        target = mimic_data_score[[score_name]]

        # LoS is skewed, log transform
        target[score_name] += 1
        news2[score_name] += 1
        target[score_name] = np.log(target)
        news2[score_name] = target[score_name]

    # first two columns are hadm_id and severity score
    selected_attributes = list(mimic_data_score)[2:]
    predictors = mimic_data_score[selected_attributes]

    # gridsearch params
    parameters = {
        'hidden_layers': [[30, 20, 10], [50, 20, 10], [100, 50, 10]],
        'weight_regularization': [0., 0.0001, 0.001],
        'early_stopping': [False],
        'dropout': [0., 0.5],
        'feature_activation': [tf.nn.tanh, tf.nn.sigmoid],
        'ranking_activation': [tf.nn.tanh, tf.nn.sigmoid]
    }

    # loss function
    nDCGScorer10 = partial(nDCG_cls, at=10)
    scoring = {'NDGC@10': nDCGScorer10}

    # set k-fold with random seed
    kf1 = KFold(n_splits=10, random_state=15, shuffle=True)

    # results storage
    list_of_ndcg = []
    news2_ndcg_list = []
    count_k = 0
    for train_index, test_index in kf1.split(predictors):
        count_k += 1

        # split the data
        X_train = predictors.iloc[train_index]
        X_test = predictors.iloc[test_index]
        y_train = target.iloc[train_index]
        y_test = target.iloc[test_index]

        # array format for dr, df version for sort function
        train_x_i = X_train.values
        train_y_i = y_train.values
        test_x_i = X_test.values
        test_y_i = y_test.values

        # split NEWS2 data accordingly, save split and compute NDCG
        news2_test = news2.iloc[test_index]
        ordered_news2 = news2_test.sort_values('news2_score', ascending=False).reset_index(drop=True)
        ordered_news2['rank'] = range(1, len(ordered_news2) + 1)

        # change path for ordered patients result
        ordered_news2.to_csv('results/rank_results_news2_fold{}.csv'.format(count_k), index=False)

        news2_ordered_scores = ordered_news2[score_name].values.tolist()
        ndcg_news2 = nDCG_(news2_ordered_scores, at=10)
        news2_ndcg_list.append(ndcg_news2)

        # scale features/ pre-process
        std_scaler = StandardScaler()
        std_scaler.fit(train_x_i)
        train_x_i = std_scaler.transform(train_x_i)

        test_x_i = std_scaler.transform(test_x_i)

        train_id_i = np.array([X_train.index.values]).transpose()
        train_all_i = np_append([train_id_i, train_x_i, train_y_i])
        #
        test_id_i = np.array([X_test.index.values]).transpose()
        test_all_i = np_append([test_id_i, test_x_i, test_y_i])

        # define ranker
        dr_w = directRanker(
            feature_activation=tf.nn.tanh,
            ranking_activation=tf.nn.tanh,
            max_steps=5000,
            print_step=50,
            start_batch_size=3,
            end_batch_size=32,
            start_qids=20,
            end_qids=100,
            feature_bias=True,
            hidden_layers=[100, 50, 10]
            # early_stopping=True
        )

        clf = GridSearchCV(dr_w, parameters, cv=5, n_jobs=-1, verbose=1, scoring=scoring, refit='NDGC@10',
                           return_train_score=False)
        clf.fit(train_x_i, train_y_i, ranking=False)
        best_estimator = clf.best_estimator_

        print("Best Parameters:")
        print(clf.best_params_)
        best = pd.DataFrame(clf.best_params_)
        best.to_csv('results/best_params_{}_fold{}.csv'.format(col_list[1], count_k))

        # get feature importance
        compare_net_p = partial(compare_net, dr_net=best_estimator)

        score = nDCGScorer10(best_estimator, test_x_i, test_y_i)

        print("Test on Fold" + str(count_k) + ": NDCG@10=" + str(score))

        list_of_ndcg.append(score[0])

        # sorted test data
        sorted_test = sorted(test_all_i, key=cmp_to_key(compare_net_p))
        sorted_df = pd.DataFrame(sorted_test)
        sorted_df[0] = sorted_df[0].astype(int)
        sorted_df = sorted_df.set_index(sorted_df[0])
        sorted_df['rank'] = range(len(sorted_df), 0, -1)

        # save results
        sorted_df.to_csv('results/rank_results_{}_fold{}.csv'.format(col_list[1], count_k), index=False)

    print(list_of_ndcg)
    df = pd.DataFrame(
        {'ndcg10 DirectRanker': list_of_ndcg, 'NDCG@10 NEWS2': news2_ndcg_list, 'Fold': list(range(1, 11))})
    df.to_csv('results/ndcg10_results_{}.csv'.format(col_list[1]), index=False)