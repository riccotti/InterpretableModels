__author__ = "Riccardo Guidotti"

import os
import datetime
import argparse

import numpy as np

import models
import stability
import datamanager
import preprocessing as prep

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import logging

import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from sklearn.metrics import f1_score


def predict_sklearn_decision_tree(clf, train_test, features):
    X_train, X_test, y_train, y_test, fsindexes = train_test
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    f1train = f1_score(y_train, y_pred_train, average='weighted')
    f1test = f1_score(y_test, y_pred_test, average='weighted')
    return f1train, f1test


def predict_yadt_decision_tree(clf, train_test, features):
    X_train, X_test, y_train, y_test, fsindexes = train_test
    dv = np.argmax(np.bincount(y_train))
    y_pred_train = clf.ypredict(X_train, targetname='class', features=features, default_value=dv)
    y_pred_test = clf.ypredict(X_test, targetname='class', features=features, default_value=dv)
    f1train = f1_score(y_train, y_pred_train, average='weighted')
    f1test = f1_score(y_test, y_pred_test, average='weighted')
    return f1train, f1test


def predict_linear_regression(clf, train_test, features):
    X_train, X_test, y_train, y_test, fsindexes = train_test
    y_pred_train = np.round(clf.predict(X_train)).astype(int)
    y_pred_test = np.round(clf.predict(X_test)).astype(int)
    f1train = f1_score(y_train, y_pred_train, average='weighted')
    f1test = f1_score(y_test, y_pred_test, average='weighted')
    return f1train, f1test


def predict_lasso(clf, train_test, features):
    X_train, X_test, y_train, y_test, fsindexes = train_test
    y_pred_train = np.round(clf.predict(X_train)).astype(int)
    y_pred_test = np.round(clf.predict(X_test)).astype(int)
    f1train = f1_score(y_train, y_pred_train, average='weighted')
    f1test = f1_score(y_test, y_pred_test, average='weighted')
    return f1train, f1test


def predict_ridge(clf, train_test, features):
    X_train, X_test, y_train, y_test, fsindexes = train_test
    y_pred_train = np.round(clf.predict(X_train)).astype(int)
    y_pred_test = np.round(clf.predict(X_test)).astype(int)
    f1train = f1_score(y_train, y_pred_train, average='weighted')
    f1test = f1_score(y_test, y_pred_test, average='weighted')
    return f1train, f1test


def predict_cpar(clf, train_test, features):
    X_train, X_test, y_train, y_test, fsindexes = train_test
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    f1train = f1_score(y_train, y_pred_train, average='weighted')
    f1test = f1_score(y_test, y_pred_test, average='weighted')
    return f1train, f1test


def predict_foil(clf, train_test, features):
    X_train, X_test, y_train, y_test, fsindexes = train_test
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    f1train = f1_score(y_train, y_pred_train, average='weighted')
    f1test = f1_score(y_test, y_pred_test, average='weighted')
    return f1train, f1test


predict_model = {
    ('DT', 'sklearn'): predict_sklearn_decision_tree,
    ('DT', 'yadt'): predict_yadt_decision_tree,
    ('LM', 'linear_regression'): predict_linear_regression,
    ('LM', 'lasso'): predict_lasso,
    ('LM', 'ridge'): predict_ridge,
    ('RB', 'cpar'): predict_cpar,
    ('RB', 'foil'): predict_foil,
}


aggregation_functions = {
    'mean': np.mean,
    'std': np.std,
    'median': np.median,
    'max': np.max,
    'min': np.min,
    '10p': lambda x: np.percentile(x, 10),
    '25p': lambda x: np.percentile(x, 25),
    '75p': lambda x: np.percentile(x, 75),
    '90p': lambda x: np.percentile(x, 90),
}


def set_argparser():
    parser = argparse.ArgumentParser(description='Evaluate interpretable models stability.')
    parser.add_argument('-d', dest='dataset_name', help='Dataset name')
    parser.add_argument('-m', dest='model_name', help='Model name')
    parser.add_argument('-dp', dest='datasets_path', help='Datasets path', default='')
    parser.add_argument('-mp', dest='models_path', help='Models path', default='')
    parser.add_argument('-rp', dest='results_path', help='Results path', default='')
    parser.add_argument('-s', dest='nbr_splits', help='Number of splits', default=10, type=int)
    parser.add_argument('-i', dest='nbr_iter', help='Number of iterations', default=5, type=int)
    parser.add_argument('-l', dest='load_model', help='Load model', default=False, action='store_true')
    parser.add_argument('-v', dest='verbose', help='Show log', default=False, action='store_true')
    parser.add_argument('-sd', dest='sd', help='Show available datasets', default=False, action='store_true')
    parser.add_argument('-sm', dest='sm', help='Show available models', default=False, action='store_true')

    return parser


def run_evaluate_overfitting(dataset_name, model_name, datasets_path='', models_path='', results_path='',
                             nbr_splits=10, nbr_iter=5, load_model=True, verbose=False):

    if verbose:
        print(datetime.datetime.now(), 'Loading %s dataset' % dataset_name)

    logging.info('%s Loading %s dataset' % (datetime.datetime.now(), dataset_name))

    X, y, e, f = datamanager.get_dataset(dataset_name, datasets_path)
    features = f
    encoder = e
    # print(features)
    # return -1

    if verbose:
        print(datetime.datetime.now(), 'Building preprocessing pipe')

    logging.info('%s Building preprocessing pipe' % datetime.datetime.now())

    preprocessing_pipe = prep.build_preprocessing_pipe()
    # print(len(preprocessing_pipe))
    # return -1

    if verbose:
        print(datetime.datetime.now(), 'Loading models')

    logging.info('%s Loading models' % datetime.datetime.now())

    trained_model = stability.load_model(model_name, dataset_name, models_path)

    # print(list(trained_model.keys())[:5])
    # print(list(trained_model[(None, ('FS', 'SelectKBest'))].keys())[:5])
    # print(trained_model[(None, ('FS', 'SelectKBest'))][(0,0,0)])
    # return -1

    if verbose:
        print(datetime.datetime.now(), 'Evaluating overfitting')

    logging.info('%s Evaluating overfitting' %datetime.datetime.now())

    overfitting = defaultdict(lambda: defaultdict(list))
    for pipe in trained_model:
        if pipe not in preprocessing_pipe:
            continue
        for fid in trained_model[pipe]:

            clf, acc, f1, fs = trained_model[pipe][fid]
            # print(np.sum(fs))

            iter_id, k, fold_id = fid
            # print(iter_id, k, fold_id)
            skf = StratifiedKFold(n_splits=nbr_splits, random_state=iter_id, shuffle=True)
            indexes = list(skf.split(X, y))[k]

            train_index, test_index = indexes
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_test = (X_train, X_test, y_train, y_test)

            if verbose:
                print(datetime.datetime.now(), '\t\t\tPreprocessing %s' % str(pipe))

            logging.info('%s\t\t\tPreprocessing %s' % (datetime.datetime.now(), str(pipe)))

            ctt = stability.copy_train_test(train_test)
            if model_name in [('DT', 'yadt')]:
                ctt, les = stability.encode_dataset(ctt, features)
                ctt, fs = prep.preprocessing(ctt, pipe, iter_id, features)
                ctt = stability.decode_dataset(ctt, les, features, fs)
            else:
                ctt, fs = prep.preprocessing(ctt, pipe, iter_id, features)
            train_test_eval = ctt[0], ctt[1], ctt[2], ctt[3], fs

            if verbose:
                print(datetime.datetime.now(), '\t\t\tPredict % s' % str(model_name))

            logging.info('%s\t\t\tPredict % s' % (datetime.datetime.now(), str(model_name)))

            f1train, f1test = predict_model[model_name](clf, train_test_eval, [f for f,s in zip(features,fs) if s])

            overfitting[pipe][(iter_id, k, fold_id)] = [f1train, f1test]


    over_eval = defaultdict(lambda: defaultdict(dict))
    for pipe in trained_model:
        if pipe not in preprocessing_pipe:
            continue
        measure_values = defaultdict(list)
        for fid in trained_model[pipe]:
            f1train, f1test = overfitting[pipe][fid]
            measure_values['f1train'].append(f1train)
            measure_values['f1test'].append(f1test)
            measure_values['diff'].append(f1train - f1test)
            measure_values['diff_norm'].append((f1train - f1test)/f1train)
            measure_values['ratio'].append(f1test / f1train)

        for measure, values in measure_values.items():
            for af_name in sorted(aggregation_functions):
                fun = aggregation_functions[af_name]
                over_eval[pipe][measure][af_name] = fun(values)

    logging.info('%s Storing results' % datetime.datetime.now())
    if verbose:
        print(datetime.datetime.now(), 'Storing results')

        store_overfitting(over_eval, aggregation_functions, model_name, dataset_name, results_path)

    if verbose:
        print(datetime.datetime.now(), 'Evaluation completed')

    logging.info('%s Evaluation completed' % datetime.datetime.now())


def store_overfitting(overfitting_eval, aggregation_functions, model_name, dataset_name, path):
    mtype, mname = model_name
    filename = '%s_%s_%s' % (dataset_name, mtype, mname)
    res_file = open(path + filename + '_res.csv', 'w')
    pipe0 = list(overfitting_eval.keys())[0]
    h1 = sorted(overfitting_eval[pipe0].keys())
    af = sorted(aggregation_functions.keys())

    sh1 = ';'.join(['%s_%s' % (h, a) for h in sorted(h1) for a in sorted(af)])

    res_file.write('dataset;model_name;preprocessing;%s\n' % sh1)

    for pipe in overfitting_eval:
        values = list()
        for m in sorted(overfitting_eval[pipe]):
            for a in sorted(overfitting_eval[pipe][m]):
                values.append(overfitting_eval[pipe][m][a])
        for m in sorted(overfitting_eval[pipe]):
            for a in sorted(overfitting_eval[pipe][m]):
                values.append(overfitting_eval[pipe][m][a])
        sval = ';'.join([str(x) for x in values])
        res_val = '%s;%s;%s;%s\n' % (dataset_name, model_name, pipe, sval)
        res_file.write(res_val)
    res_file.close()


def main():

    parser = set_argparser()
    args = parser.parse_args()

    if args.sd:
        for d in sorted(datamanager.datasets):
            print(d)
        return 0

    if args.sm:
        for m in sorted(predict_model):
            print(m)
        return 0

    dataset_name = args.dataset_name
    model_name = eval(str(args.model_name))
    datasets_path = args.datasets_path
    models_path = args.models_path
    results_path = args.results_path

    if dataset_name is None or model_name is None:
        print(parser.print_help())
        return 1

    nbr_splits = args.nbr_splits
    nbr_iter = args.nbr_iter
    load_model = args.load_model
    verbose = args.verbose

    logfile = '../logs/%s_%s_%s.log' % (dataset_name.split('_')[0], model_name[0], model_name[1])
    if logfile and os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(format='%(message)s', filename=logfile, level=logging.INFO)

    print(dataset_name, model_name, datasets_path, models_path, results_path,
                             nbr_splits, nbr_iter, load_model, verbose)

    run_evaluate_overfitting(dataset_name, model_name, datasets_path, models_path, results_path,
                             nbr_splits, nbr_iter, load_model, verbose)


if __name__ == "__main__":
    main()
