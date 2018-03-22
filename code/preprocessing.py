from sklearn.feature_selection import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

feature_selection = {
    'VarianceThreshold': (VarianceThreshold,),
    # 'SelectKBest': (SelectKBest,),
    'SelectPercentile': (SelectPercentile,),
    'SelectFpr': (SelectFpr,),
    'RFE_LogisticRegression': (RFE, LogisticRegression,),
    'SelectFromModel_LinearSVC': (SelectFromModel, LinearSVC,),
    'SelectFromModel_ExtraTreesClassifier': (SelectFromModel, ExtraTreesClassifier,),
}

instance_selection = {
    'RandomOverSampler': RandomOverSampler,
    'ADASYN': ADASYN,
    'SMOTE': SMOTE,
    'RandomUnderSampler': RandomUnderSampler,
    'CondensedNearestNeighbour': CondensedNearestNeighbour,
    'ClusterCentroids': ClusterCentroids,
    'SMOTEENN': SMOTEENN,
}


def preprocessing(train_test, pipe, seed):
    X_train, X_test, y_train, y_test = train_test
    for step in pipe:

        if step is None:
            continue

        if step[0] == 'FS':
            fs_name = step[1]
            fs = feature_selection[fs_name]
            feat_sel = fs[0]() if len(fs) == 1 else fs[0](fs[1](random_state=seed))
            X_train = feat_sel.fit_transform(X_train, y_train)
            X_test = feat_sel.transform(X_test)
            continue

        if step[0] == 'IS':
            is_name = step[1]
            inst_sel = instance_selection[is_name](random_state=seed)
            X_train, y_train = inst_sel.fit_sample(X_train, y_train)
            continue

    return X_train, X_test, y_train, y_test


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
