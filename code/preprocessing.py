__author__ = "Riccardo Guidotti"

import numpy as np

from sklearn.feature_selection import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from sklearn.tree import DecisionTreeClassifier


# from imblearn.combine import *
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")


feature_selection = {
    # 'VarianceThreshold': (VarianceThreshold,),
    'SelectKBest': (SelectKBest,),
    'SelectPercentile': (SelectPercentile,),
    'RFE_DecisionTreeClassifier': (RFE, DecisionTreeClassifier),

    # 'SelectFpr': (SelectFpr,),
    # 'RFE_LogisticRegression': (RFE, LogisticRegression,),
    # 'SelectFromModel_LinearSVC': (SelectFromModel, LinearSVC,),
    # 'SelectFromModel_ExtraTreesClassifier': (SelectFromModel, ExtraTreesClassifier,),
}

instance_selection = {
    'RandomOverSampler': RandomOverSampler,
    'SMOTE': SMOTE,
    'RandomUnderSampler': RandomUnderSampler,

    # 'ClusterCentroids': ClusterCentroids,
    # 'CondensedNearestNeighbour': CondensedNearestNeighbour,
    # 'ADASYN': ADASYN,
    # 'SMOTEENN': SMOTEENN,
}


def fix_integer_columns(X, features, fsindexes):
    X0 = None
    i = 0
    for v, fu in zip(features, fsindexes):
        col, col_type, feat_type = v
        if fu:
            X_col = X[:, i]
            if col_type == 'integer' or col_type == 'string':
                X_col = X_col.astype(int)
            X0 = X_col if X0 is None else np.column_stack((X0, X_col))
            i += 1
    return X0


def preprocessing(train_test, pipe, seed, features):
    X_train, X_test, y_train, y_test = train_test
    fsindexes = [True] * X_train.shape[1]
    for step in pipe:

        if step is None:
            continue

        if step[0] == 'FS':
            fs_name = step[1]
            fs = feature_selection[fs_name]
            feat_sel = fs[0]() if len(fs) == 1 else fs[0](fs[1](random_state=seed))

            X_train = feat_sel.fit_transform(X_train, y_train)
            X_test = feat_sel.transform(X_test)
            fsindexes = feat_sel.get_support()

            continue

        if step[0] == 'IS':
            is_name = step[1]
            inst_sel = instance_selection[is_name](random_state=seed)
            X_train, y_train = inst_sel.fit_sample(X_train, y_train)
            if step[1] in ['SMOTE', 'ADASYN', 'ClusterCentroids']:
                X_train = fix_integer_columns(X_train, features, fsindexes)
                # X_test = fix_integer_columns(X_test, features, fsindexes)

            continue

    return (X_train, X_test, y_train, y_test), fsindexes


def build_preprocessing_pipe():
    preprocessing_pipe = [(None, None)]
    for fsm in feature_selection:
        preprocessing_pipe.append((None, ('FS', fsm)))
    for ism in instance_selection:
        preprocessing_pipe.append((None, ('IS', ism)))
    for fsm in feature_selection:
        for ism in instance_selection:
            preprocessing_pipe.append((('FS', fsm), ('IS', ism)))
    for ism in instance_selection:
        for fsm in feature_selection:
            preprocessing_pipe.append((('IS', ism), ('FS', fsm)))

    return preprocessing_pipe
