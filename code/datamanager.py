__author__ = "Riccardo Guidotti"

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from discretize import QuartileDiscretizer
from discretize import DecileDiscretizer
from discretize import EntropyDiscretizer

import warnings
warnings.filterwarnings("ignore")


def prepare_iris_sklearn(dataset_name, dataset_path):
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


def get_features(filename):
    data = open(filename, 'r')
    features = list()
    feature_names = list()
    usecols = list()
    col_id = 0
    for row in data:
        field = row.strip().split(',')
        feature_names.append(field[0])
        if field[2] != 'ignore':
            usecols.append(col_id)
            if field[2] != 'class':
                features.append((field[0], field[1], field[2]))
        col_id += 1
    return feature_names, features, usecols


def prepare_sklearn_dataset(dataset_name, dataset_path, target='class'):
    # df = pd.read_csv(dataset_path + dataset_name + '.csv.gz', delimiter=',')

    feature_names, features, col_indexes = get_features(dataset_path + dataset_name + '.names')
    df = pd.read_csv(dataset_path + dataset_name + '.data.gz', delimiter=',', names=feature_names, usecols=col_indexes)

    # features = yadt.metadata(df, ovr_types={})

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
        f = sklearn_metadata(features, e.feature_indices_, features2binarize, X.shape[1], target)
    else:
        e = None
        f = features

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

    feature_names, features, col_indexes = get_features(dataset_path + dataset_name + '.names')
    df = pd.read_csv(dataset_path + dataset_name + '.data.gz', delimiter=',', names=feature_names, usecols=col_indexes)

    X = df.loc[:, df.columns != target].values
    y = df[target].values

    return X, y, None, features


def prepare_rule_dataset(dataset_name, dataset_path, target='class', discretizer='entropy'):

    feature_names, features, col_indexes = get_features(dataset_path + dataset_name + '.names')
    df = pd.read_csv(dataset_path + dataset_name + '.data.gz', delimiter=',', names=feature_names, usecols=col_indexes)

    X = df.loc[:, df.columns != target].values
    y = df[target].values

    # step 1: discretize continuous attributes
    features2discretize = list()
    for col, col_type, feat_type in features:
        if feat_type == 'continuous':
            features2discretize.append(col)

    if len(features2discretize) > 0:
        categorical_features = [i for i in range(0, len(features)) if features[i][0] not in features2discretize]
        feature_names_disc = [col for col, col_type, feat_type in features]

        if discretizer == 'quartile':
            discretizer = QuartileDiscretizer(X, categorical_features, feature_names_disc, labels=y)
        elif discretizer == 'decile':
            discretizer = DecileDiscretizer(X, categorical_features, feature_names_disc, labels=y)
        elif discretizer == 'entropy':
            discretizer = EntropyDiscretizer(X, categorical_features, feature_names_disc, labels=y)
        else:
            raise ValueError("Discretizer must be 'quartile', 'decile' or 'entropy'")
        # cont_features = [i for i in range(0, len(features)) if i not in categorical_features]
        X = discretizer.discretize(X)

    # step 2: label encode
    Xy = np.column_stack((X, y))
    df = pd.DataFrame(data=Xy, columns=[c[0] for c in features] + [target])
    nbr_last_values = 0
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col]) + nbr_last_values
        nbr_last_values += len(df[col].unique())

    X = df.loc[:, df.columns != target].values
    y = df[target].values

    features = [(col, 'integer', 'discrete') for col, col_type, feat_type in features]

    return X, y, None, features


dataset_names = ['credit', 'adult', 'cover', 'coil2000', 'clean1', 'clean2', 'gisette', 'isolet', 'madelon',
                 'credit_small']

datasets = dict()

for dn in dataset_names:
    datasets['%s_sklearn' % dn] = prepare_sklearn_dataset

for dn in dataset_names:
    datasets['%s_yadt' % dn] = prepare_yadt_dataset

for dn in dataset_names:
    datasets['%s_rule' % dn] = prepare_rule_dataset


def get_dataset(dataset_name, dataset_path):

    return datasets[dataset_name](dataset_name.split('_')[0], dataset_path)


