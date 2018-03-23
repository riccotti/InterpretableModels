__author__ = "Riccardo Guidotti"

from yadt import yadt

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def prepare_iris_sklearn(dataset_name, dataset_path):
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


def get_features(filename):
    data = open(filename, 'r')
    features = list()
    usecols = list()
    col_id = 0
    for row in data:
        field = row.strip().split(',')
        features.append(field[0])
        if field[2] != 'ignore':
            usecols.append(col_id)
        col_id += 1
    return features, usecols


def prepare_sklearn_dataset(dataset_name, dataset_path, target='class'):
    # df = pd.read_csv(dataset_path + dataset_name + '.csv.gz', delimiter=',')

    features, col_indexes = get_features(dataset_path + dataset_name + '.names')
    df = pd.read_csv(dataset_path + dataset_name + '.data.gz', delimiter=',', names=features, usecols=col_indexes)

    features = yadt.metadata(df, ovr_types={})

    features2binarize = list()
    class_observed = False
    for idx, col in enumerate(df.columns):
        dtype = df[col].dtype
        if dtype != np.float64:
            if dtype.kind == 'O':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                if len(le.classes_) > 2:
                    # print(col, idx if not class_observed else idx - 1)
                    features2binarize.append(idx if not class_observed else idx - 1)
        if col == target:
            class_observed = True

    X = df.loc[:, df.columns != target].values
    y = df[target].values

    if len(features2binarize) > 1:
        e = OneHotEncoder(categorical_features=features2binarize)
        X = e.fit_transform(X).toarray()
    else:
        e = None

    f = sklearn_metadata(features, e.feature_indices_, features2binarize, X.shape[1], target)

    return X, y, e, f


def sklearn_metadata(f, fi, fb, n, t):

    res = [0] * (n + 1)
    m = fi[-1]
    idx_fnb = 0  # indice features not binarize
    idx_fb = 0
    i = 0
    class_observed = 0
    for col, col_type, feat_type in f:
        if col == t:
            res[i] = (col, col_type, feat_type)
            i -= 1
            class_observed = 1
        else:
            if i not in fb:
                res[m + idx_fnb + class_observed] = (col, col_type, feat_type)
                idx_fnb += 1
            else:
                for j in range(fi[idx_fb], fi[idx_fb+1]):
                    res[j + class_observed] = (col, col_type, feat_type)
                idx_fb += 1
        i += 1
    return res


def prepare_yadt_dataset(dataset_name, dataset_path, target='class'):

    features, col_indexes = get_features(dataset_path + dataset_name + '.names')
    df = pd.read_csv(dataset_path + dataset_name + '.data.gz', delimiter=',', names=features, usecols=col_indexes)

    features = yadt.metadata(df, ovr_types={})

    X = df.loc[:, df.columns != target].values
    y = df[target].values

    return X, y, None, features


datasets = {
    'iris_sklearn': prepare_iris_sklearn,
    'credit_small_sklearn': prepare_sklearn_dataset,

    'credit_sklearn': prepare_sklearn_dataset,
    'adult_sklearn': prepare_sklearn_dataset,
    'cover_sklearn': prepare_sklearn_dataset,
    'coil2000_sklearn': prepare_sklearn_dataset,
    'clean1_sklearn': prepare_sklearn_dataset,
    'clean2_sklearn': prepare_sklearn_dataset,
    'gisette_sklearn': prepare_sklearn_dataset,
    'isolet_sklearn': prepare_sklearn_dataset,
    'madelon_sklearn': prepare_sklearn_dataset,

    'credit_yadt': prepare_yadt_dataset,
    'adult_yadt': prepare_yadt_dataset,
    'cover_yadt': prepare_yadt_dataset,
    'coil2000_yadt': prepare_yadt_dataset,
    'clean1_yadt': prepare_yadt_dataset,
    'clean2_yadt': prepare_yadt_dataset,
    'gisette_yadt': prepare_yadt_dataset,
    'isolet_yadt': prepare_yadt_dataset,
    'madelon_yadt': prepare_yadt_dataset,
}


def get_dataset(dataset_name, dataset_path):

    return datasets[dataset_name](dataset_name.split('_')[0], dataset_path)
