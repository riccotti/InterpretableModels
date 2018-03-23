__author__ = "Riccardo Guidotti"

import datetime
import numpy as np
import _pickle as cPickle

import preprocessing as prep

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold


def copy_train_test(train_test):
    X_train, X_test, y_train, y_test = train_test
    X_train, X_test, y_train, y_test = X_train[:], X_test[:], y_train[:], y_test[:]
    return X_train, X_test, y_train, y_test


def train_model(X, y, model_name, preprocessing_pipe, fit_predict_model, features, nbr_splits=10, nbr_iter=5,
                verbose=False):
    trained_model = defaultdict(lambda: defaultdict(list))
    for iter_id in range(0, nbr_iter):
        if verbose:
            print(datetime.datetime.now(), '\tIteration % d' % iter_id)
        skf = StratifiedKFold(n_splits=nbr_splits, random_state=iter_id, shuffle=True)
        for k, indexes in enumerate(skf.split(X, y)):
            if verbose:
                print(datetime.datetime.now(), '\t\tFold % d' % k)
            train_index, test_index = indexes
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_test = (X_train, X_test, y_train, y_test)
            for pipe in preprocessing_pipe:
                if verbose:
                    print(datetime.datetime.now(), '\t\t\tPreprocessing % s' % str(pipe))
                train_test_eval = prep.preprocessing(copy_train_test(train_test), pipe, iter_id, features)
                maf = fit_predict_model[model_name](train_test_eval, iter_id)
                fold_id = len(trained_model[pipe])
                trained_model[pipe][(iter_id, k, fold_id)] = maf

    return trained_model


def store_model(trained_model, model_name, dataset_name, path):
    mtype, mname = model_name
    filename = '%s_%s_%s' % (dataset_name, mtype, mname)
    text_file = open(path + filename + '.csv', 'w')
    model_file = open(path + filename + '.pckl', 'wb')
    text_file.write('dataset;model_name;preprocessing;iter_id;k;fold_id;acc_score;f1_score,features\n')

    for pipe in trained_model:
        for fid in trained_model[pipe]:
            iter_id, k, fold_id = fid
            model, acc, f1, fs = trained_model[pipe][fid]
            cPickle.dump(model, model_file)
            text_file.write('%s;%s;%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, model_name, pipe, iter_id, k, fold_id, acc, f1, ' '.join(str(x) for x in fs.tolist())))
        text_file.flush()
    model_file.close()
    text_file.close()


def load_model(model_name, dataset_name, path):
    mtype, mname = model_name
    filename = '%s_%s_%s' % (dataset_name, mtype, mname)

    model_file = open(path + filename + '.pckl', 'rb')
    models = list()
    while True:
        try:
            models.append(cPickle.load(model_file))
        except EOFError:
            break
    model_file.close()

    text_file = open(path + filename + '.csv', 'r')
    text_file.readline()

    trained_model = defaultdict(lambda: defaultdict(list))
    for line in text_file:
        fields = line.strip().split(';')
        pipe = eval(str(fields[2]))
        iter_id = int(fields[3])
        k = int(fields[4])
        fold_id = int(fields[5])
        acc = float(fields[6])
        f1 = float(fields[7])
        fs = np.array([bool(x) for x in fields[8].split(' ')])
        trained_model[pipe][(iter_id, k, fold_id)] = (models[fold_id], acc, f1)

    text_file.close()
    return trained_model


def defaultdict2dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict2dict(v) for k, v in d.items()}
    return d


def evaluate_model_stability(model_name, trained_model, analyze_model, aggregation_functions, encoder):
    prep_measure_agg_value = defaultdict(lambda: defaultdict(dict))
    for pipe in trained_model:
        measure_values = defaultdict(list)
        for fid in trained_model[pipe]:
            model, acc, f1, fs = trained_model[pipe][fid]
            meval = analyze_model[model_name](model, encoder, fs)

            for measure in meval:
                measure_values[measure].append(meval[measure])
            measure_values['accuracy'].append(acc)
            measure_values['f1score'].append(f1)

        for measure, values in measure_values.items():
            for af_name in sorted(aggregation_functions):
                fun = aggregation_functions[af_name]
                prep_measure_agg_value[pipe][measure][af_name] = fun(values)

    return defaultdict2dict(prep_measure_agg_value)

    # prep_measure_agg_value = defaultdict(lambda: defaultdict(dict))
    # for pipe in trained_model:
    #     measure_values = defaultdict(list)
    #     for fid in trained_model[pipe]:
    #         model, acc, f1 = trained_model[pipe][fid]
    #         meval = analyze_model[model_name](model)
    #
    #         for measure in meval:
    #             measure_values[measure].append(meval[measure])
    #         measure_values['accuracy'].append(acc)
    #         measure_values['f1score'].append(f1)
    #
    #     for measure, values in measure_values.items():
    #         for af_name in sorted(aggregation_functions):
    #             fun = aggregation_functions[af_name]
    #             prep_measure_agg_value[pipe][measure][af_name] = fun(values)
    #
    # return defaultdict2dict(prep_measure_agg_value)


def evaluate_model_stability_comparison(model_name, trained_model, compare_models, aggregation_functions, encoder):
    prep_measure_agg_value = defaultdict(lambda: defaultdict(dict))
    for pipe in trained_model:
        measure_values = defaultdict(list)
        tm_list = sorted(trained_model[pipe].keys())
        brackets_repository = dict()
        for i in range(0, len(tm_list)):
            for j in range(i + 1, len(tm_list)):
                m1, _, _, fs1 = trained_model[pipe][tm_list[i]]
                m2, _, _, fs2 = trained_model[pipe][tm_list[j]]
                args = (i, j, brackets_repository)
                meval = compare_models[model_name](m1, m2, encoder, fs1, fs2, args)

                for measure in meval:
                    measure_values[measure].append(meval[measure])

        for measure, values in measure_values.items():
            for af_name in sorted(aggregation_functions):
                fun = aggregation_functions[af_name]
                prep_measure_agg_value[pipe][measure][af_name] = fun(values)
    return defaultdict2dict(prep_measure_agg_value)


def store_model_stability(model_stability, model_stability_comparison, aggregation_functions,
                          model_name, dataset_name, path):
    mtype, mname = model_name
    filename = '%s_%s_%s' % (dataset_name, mtype, mname)
    res_file = open(path + filename + '_res.csv', 'w')
    pipe0 = list(model_stability.keys())[0]
    h1 = sorted(model_stability[pipe0].keys())
    h2 = sorted(model_stability_comparison[pipe0].keys())
    af = sorted(aggregation_functions.keys())

    sh1 = ';'.join(['%s_%s' % (h, a) for h in sorted(h1) for a in sorted(af)])
    sh2 = ';'.join(['%s_%s' % (h, a) for h in sorted(h2) for a in sorted(af)])

    res_file.write('dataset;model_name;preprocessing;%s;%s\n' % (sh1, sh2))

    for pipe in model_stability:
        values = list()
        for m in sorted(model_stability[pipe]):
            for a in sorted(model_stability[pipe][m]):
                values.append(model_stability[pipe][m][a])
        for m in sorted(model_stability_comparison[pipe]):
            for a in sorted(model_stability_comparison[pipe][m]):
                values.append(model_stability_comparison[pipe][m][a])
        sval = ';'.join([str(x) for x in values])
        res_val = '%s;%s;%s;%s\n' % (dataset_name, model_name, pipe, sval)
        res_file.write(res_val)
    res_file.close()

