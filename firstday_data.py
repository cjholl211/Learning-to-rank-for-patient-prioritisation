import pandas as pd
from functools import reduce
from rf_helpers import one_hot_encode

# data taken from MIMIC-III
col_list = ['hadm_id', 'admission_type', 'gender', 'age']
demographic_data = pd.read_csv("demographic_data.csv", 'r', delimiter=",", usecols=col_list)

# filter those under 18
demographic_data = demographic_data[demographic_data.age >= 18]

# vitals
col_list = ['hadm_id', 'heartrate_mean', 'sysbp_mean', 'tempc_mean', 'resprate_mean', 'spo2_mean', 'diasbp_mean', 'meanbp_mean']
vitals = pd.read_csv("vitalsfirstday.csv", 'r', usecols=col_list, delimiter=',')


col_list = ['hadm_id','bicarbonate_mean', 'bun_mean', 'wbc_mean', 'aniongap_mean', 'platelet_mean']
labs = pd.read_csv("labs_firstday_mean.csv",
                   'r', usecols=col_list, delimiter=',')

# gcs
col_list = ['hadm_id', 'mingcs']
gca = pd.read_csv("gcsfirstday.csv", 'r', usecols=col_list, delimiter=',')

# urine
col_list = ['hadm_id', 'urineoutput']
urine = pd.read_csv("uofirstday.csv", 'r', usecols=col_list, delimiter=',')

col_list = ['hadm_id', 'vent']
vent = pd.read_csv("ventfirstday.csv", 'r',  usecols=col_list, delimiter=',')

# merge all dataframes
data_frames = [gca, vent, urine, labs, vitals]

df = reduce(lambda left, right: pd.merge(left, right, on=['hadm_id'],
                                         how='outer'), data_frames)

# repeat recordings, take the mean
df = df.groupby('hadm_id').mean()

df = pd.merge(df, demographic_data, how='inner', on='hadm_id')

# remove na rows
df = df.dropna()

# remove elective admissions
df = df[df.admission_type != 'ELECTIVE']

# bin and one hot encode GCS
bins = [0, 8, 12, 15]
labels = ['severe_gcs', 'moderate_gcs', 'mild_gcs']
df['mingcs_bins'] = pd.cut(df.mingcs, bins, labels=labels)

# one hot encode categorical data
cat_data = ['gender', 'mingcs_bins']
for i in cat_data:
    df_temp = one_hot_encode(df, i)
    df_temp.reset_index()
    df = pd.merge(df, df_temp, on='hadm_id', how='inner')


df = df.drop(['admission_type', 'gender', 'mingcs_bins'], axis=1)

# remove rows of ages over 300
df = df[df['age'] < 300]

df.to_csv('unscaled_ltr_data.csv',
          index=False)
