import pandas as pd
import numpy as np

col_list = ['hadm_id', 'vent', 'resprate_mean', 'tempc_mean', 'sysbp_mean', 'heartrate_mean', 'spo2_mean']
news_data = pd.read_csv('unscaled_ltr_data.csv', 'r', delimiter=',', usecols=col_list)

col_list = ['hadm_id', 'mingcs']
gcs = pd.read_csv("gcsfirstday.csv", 'r', usecols=col_list, delimiter=',')

gcs = gcs.groupby('hadm_id').mean()

# data for news 2
news_data = pd.merge(news_data, gcs, how='inner', on='hadm_id')

conds = [[news_data.vent < 1, news_data.vent == 1],
         [news_data.resprate_mean <= 8, (8 < news_data.resprate_mean) & (news_data.resprate_mean <= 11),
          (11 < news_data.resprate_mean) & (news_data.resprate_mean <= 20),
          (20 < news_data.resprate_mean) & (news_data.resprate_mean <= 24), news_data.resprate_mean > 24],
         [news_data.tempc_mean <= 35, (35 < news_data.tempc_mean) & (news_data.tempc_mean <= 36),
          (36 < news_data.tempc_mean) & (news_data.tempc_mean <= 38),
          (38 < news_data.tempc_mean) & (news_data.tempc_mean <= 39), news_data.tempc_mean > 39],
         [news_data.sysbp_mean <= 90, (90 < news_data.sysbp_mean) & (news_data.sysbp_mean <= 100),
          (100 < news_data.sysbp_mean) & (news_data.sysbp_mean <= 110),
          (110 < news_data.sysbp_mean) & (news_data.sysbp_mean <= 219), news_data.sysbp_mean > 219],
         [news_data.heartrate_mean <= 40, (40 < news_data.heartrate_mean) & (news_data.heartrate_mean <= 50),
          (50 < news_data.heartrate_mean) & (news_data.heartrate_mean <= 90),
          (90 < news_data.heartrate_mean) & (news_data.heartrate_mean <= 110),
          (110 < news_data.heartrate_mean) & (news_data.heartrate_mean <= 130), news_data.heartrate_mean > 130],
         [news_data.spo2_mean <= 91, (91 < news_data.spo2_mean) & (news_data.spo2_mean <= 93),
          (93 < news_data.spo2_mean) & (news_data.spo2_mean <= 95), news_data.spo2_mean > 95],
         [news_data.mingcs < 15, news_data.mingcs == 15]]
choices = [[0, 2],
           [3, 1, 0, 2, 3],
           [3, 1, 0, 1, 2],
           [3, 2, 1, 0, 3],
           [3, 1, 0, 1, 2, 3],
           [3, 2, 1, 0],
           [3, 0]]


def news2_convert(choices, conds, news_data):
    hadm_id_df = news_data.pop('hadm_id')
    feature_list = list(news_data)
    i = -1
    for f in feature_list:
        i += 1
        data = news_data[[f]]
        scores_ = pd.DataFrame(np.select(conds[i], choices[i], default='zero'),
                               index=data.index,
                               columns=data.columns)
        scores_.rename({f: f + '_score'}, axis=1, inplace=True)
        news_data = pd.concat([news_data, scores_], axis=1)
    news_data = news_data.iloc[:, 7:]
    news_data['news2_score'] = news_data.apply(pd.to_numeric).sum(axis=1)
    news_data = pd.concat([hadm_id_df, news_data], axis=1)

    return news_data


scores = news2_convert(choices, conds, news_data)

scores.to_csv('news2_conversion_scores.csv', index=False)
