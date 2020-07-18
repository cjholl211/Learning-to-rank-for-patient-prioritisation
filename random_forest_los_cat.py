import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from rf_helpers import kfold_gridsearch_multiclass

# load in preprocessed data/ change inclusion criteria as required
#col_list = ['hadm_id', 'vent', 'heartrate_mean', 'sysbp_mean', 'spo2_mean', 'resprate_mean', 'mild_gcs', 'moderate_gcs', 'severe_gcs', 'tempc_mean']
df = pd.read_csv("unscaled_ltr_data.csv", 'r',
                 delimiter=",")#, usecols=col_list)

# get los and death
col_list = ['hadm_id', 'los', 'death']
los_death = pd.read_csv("death_within_3_days.csv", 'r',
                 delimiter=",", usecols=col_list)  # data
los_death = los_death.groupby('hadm_id').mean().round()

# place los into bins (short, medium, long)
bins = [-1, 4, 175]
names_cat = ['short', 'long']

los_death['los'] = pd.cut(los_death.los,
                  bins,
                  labels=names_cat)

df = pd.read_csv("unscaled_ltr_data.csv", 'r',
                 delimiter=",")

df = pd.merge(df, los_death, how='inner', on='hadm_id')

df.death.value_counts()
df.loc[df.death == 1, 'los'] = 'long'

df = df.drop(['death', 'hadm_id'], axis=1)

target = df[['los']]

predictors = df.drop(columns=['los'])

df.los.value_counts()

# params for grid search
parameters = {
    'max_depth': [ 2, 3, 5, 6, 7, 5, 10, 20, 30, None],
    'max_features': ['auto', 'sqrt', None],
    'min_samples_leaf': [2, 3, 4, 5, 10, 15, 20, 25]
    }

classifier = DecisionTreeClassifier(criterion='entropy')

kf1 = KFold(n_splits=5, random_state=15, shuffle=True)

# this prints best parameters, most important features and then returns avg classification report/fpr/tpr/roc auc
reports_all_data, fpr_all_data, tpr_all_data, roc_auc = kfold_gridsearch_multiclass(target, predictors, classifier,
                                                                                    parameters, kf1,
                                                                                    scoring='balanced_accuracy',
                                                                                    cv=5, roc_auc_plot=True)
# fpr/tpr/roc auc returned for any additional plots when required