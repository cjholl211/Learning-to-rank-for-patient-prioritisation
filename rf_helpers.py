from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler


# ICD-9 code pre-processing
def remove_surrounding_apostrophes(x):
    if x[0] == "'":
        x = x[1:]
    if x[-1] == "'":
        x = x[:-1]
    return x

# ICD-9 code pre-processing
def remove_apostrophes(data):
    # enter pandas df of ICD9 translations
    for c in data.columns:
        data[c] = data[c].map(remove_surrounding_apostrophes)
        idxRemove = data[c].str.strip() == ''
        if idxRemove.any():
            data.loc[idxRemove, c] = None
    return data


# # ICD-9 code pre-processing, replace codes using df of correct codes
def icd9_translations(data, translation_df):
    index_to_remove = []
    index_to_keep = []
    to_keep_values = []
    z = 1
    for i, line in data.iterrows():
        z += 1
        code = line['code']
        row = translation_df.loc[translation_df['icd9_code'] == code]
        # many icd9 codes from mimic do not match - potentially recorded incorrectly
        if row.empty:
            index_to_remove.append(i)
            continue
        if row.iloc[0][-2] is not None:
            val = row.iloc[0][-2]
            index_to_keep.append(i)
            to_keep_values.append(val)
            continue
        if row.iloc[0][-4] is not None:
            val = row.iloc[0][-4]
            index_to_keep.append(i)
            to_keep_values.append(val)
            continue
        if row.iloc[0][-6] is not None:
            val = row.iloc[0][-6]
            index_to_keep.append(i)
            to_keep_values.append(val)
            continue
        if row.iloc[0][-8] is not None:
            val = row.iloc[0][-8]
            index_to_keep.append(i)
            to_keep_values.append(val)
            continue
    # replace admission diagnosis codes (dict_data) with existing categorised codes
    data = data.iloc[index_to_keep,]

    data.insert(2, "category_codes", to_keep_values, True)

    data = data.drop('code', 1)

    return data


def one_hot_encode(data, col):
    data = data.groupby(['hadm_id'])[col].apply(list).to_frame()

    v = data[col].values
    l = [len(x) for x in v.tolist()]
    f, u = pd.factorize(np.concatenate(v))
    n, m = len(v), u.size
    i = np.arange(n).repeat(l)

    dummies = pd.DataFrame(
        np.bincount(i * m + f, minlength=n * m).reshape(n, m),
        data.index, u
    )

    data = data.merge(dummies, left_index=True, right_index=True)

    del data[col]

    return data


def roc_data(X_test, y_test, best_model, labels):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_pred = best_model.predict_proba(X_test)

    # store fpr/tpr/roc_auc in respective dictionaries

    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(labels))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(labels)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(labels)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # roc_auc_macro.append(roc_auc["macro"])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # roc_auc_micro.append(roc_auc["micro"])

    return roc_auc, fpr, tpr


def plot_roc(fpr, tpr, roc_auc, labels):
    # Plot all ROC curves
    lw = 2

    if len(labels) > 2:
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(len(labels)), colors):  # TODO
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(labels[i], roc_auc[i]))
    else:
        print('yep')
        plt.figure(figsize=(6, 6))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' Average receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, roc_auc



def kfold_gridsearch_multiclass(target, predictors, classifier, parameters, k_fold, scoring, cv, roc_auc_plot):
    count_k = 0
    reports = pd.DataFrame([])  # store classification reports
    if roc_auc_plot:
        categories = list(target['los'].unique())
        keys = [*range(len(categories))] + ['micro', 'macro']  # create dictionary keys for storing results
        fpr_avg = dict.fromkeys(keys, [])
        tpr_avg = dict.fromkeys(keys, [])

    for train_index, test_index in k_fold.split(predictors):

        count_k += 1

        X_train = predictors.iloc[train_index]
        X_test = predictors.iloc[test_index]
        y_train = target.iloc[train_index]
        y_test = target.iloc[test_index]

        names = list(X_train)
        # standardise data
        sc_X = StandardScaler()
        sc_X.fit(X_train)
        X_train = pd.DataFrame(sc_X.transform(X_train))
        X_test = pd.DataFrame(sc_X.transform(X_test))

        X_train.columns = names
        X_test.columns = names

        grid_search = GridSearchCV(classifier, parameters, n_jobs=-1, verbose=1, cv=cv, scoring=scoring)
        grid_search.fit(X_train, y_train)

        # get best model
        best_model = grid_search.best_estimator_  # best model according to grid search

        # print 'best' mdoels paramters
        best_model_parameters = best_model.get_params()
        print('k = {}'.format(count_k))
        for parameter in parameters:
            print('{:30}\t{}'.format(parameter, best_model_parameters[parameter]))
        print()

        # get feature importance
        important_features = list(zip(best_model.feature_importances_, X_test.columns.values))
        important_features.sort(reverse=True)

        # print top 20 features
        for feature_importance, feature_name in important_features[:5]:
            print('{:20s}:{:3.4f}'.format(feature_name, feature_importance))

        print('-' * 60)
        y_pred = best_model.predict(X_test)
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
        reports = reports.append(report)

        if roc_auc_plot:

            labels = best_model.classes_
            if len(labels) == 2:
                lb = MyLabelBinarizer()
                y_test = lb.fit_transform(y_test)
            else:
                y_test = label_binarize(y_test, classes=labels)  # keep the same labels

            roc_auc, fpr, tpr = roc_data(X_test, y_test, best_model, labels)
            # store lists of each iteration in the dictionary
            for i in keys:
                x = fpr_avg[i]
                y = fpr[i].tolist()
                fpr_avg[i] = x + y

                x = tpr_avg[i]
                y = tpr[i].tolist()
                tpr_avg[i] = x + y

    if roc_auc_plot:
        # average auc for each plot
        for key in keys:
            x = roc_auc[key]
            y = np.mean(x)
            roc_auc[key] = y

        # we must sort the data
        for key in keys:
            x = fpr_avg[key]
            y = tpr_avg[key]
            fpr_avg[key] = sorted(x, key=float)
            tpr_avg[key] = sorted(y, key=float)

        fpr_avg, tpr_avg, roc_auc = plot_roc(fpr_avg, tpr_avg, roc_auc, labels)

    reports = reports.groupby(reports.index).mean()
    return reports, fpr_avg, tpr_avg, roc_auc


class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((1 - Y, Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)
