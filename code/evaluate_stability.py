__author__ = "Riccardo Guidotti"

import os
import datetime
import argparse

import numpy as np

import models
import stability
import datamanager
import preprocessing as prep

import logging

import warnings
warnings.filterwarnings("ignore")


fit_predict_model = {
    ('DT', 'sklearn'): models.fit_predict_sklearn_decision_tree,
    ('DT', 'yadt'): models.fit_predict_yadt_decision_tree,
    ('LM', 'linear_regression'): models.fit_predict_linear_regression,
    ('LM', 'lasso'): models.fit_predict_lasso,
    ('LM', 'ridge'): models.fit_predict_ridge,
    ('RB', 'cpar'): models.fit_predict_cpar,
    ('RB', 'foil'): models.fit_predict_cpar,
}

analyze_model = {
    ('DT', 'sklearn'): models.analyze_sklearn_decision_tree,
    ('DT', 'yadt'): models.analyze_yadt_decision_tree,
    ('LM', 'linear_regression'): models.analyze_sklearn_linear_models,
    ('LM', 'lasso'): models.analyze_sklearn_linear_models,
    ('LM', 'ridge'): models.analyze_sklearn_linear_models,
    ('RB', 'cpar'): models.analyze_rule_based,
    ('RB', 'foil'): models.analyze_rule_based,
}

compare_models = {
    ('DT', 'sklearn'): models.compare_sklearn_decision_trees,
    ('DT', 'yadt'): models.compare_yadt_decision_trees,
    ('LM', 'linear_regression'): models.compare_sklearn_linear_models,
    ('LM', 'lasso'): models.compare_sklearn_linear_models,
    ('LM', 'ridge'): models.compare_sklearn_linear_models,
    ('RB', 'cpar'): models.compare_rule_based,
    ('RB', 'foil'): models.compare_rule_based,
}

aggregation_functions = {
    'mean': np.mean,
    'std': np.std,
    'median': np.median,
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


def run_evaluate_stability(dataset_name, model_name, datasets_path='', models_path='', results_path='',
                           nbr_splits=10, nbr_iter=5, load_model=False, verbose=False):

    # nbr_splits = 10
    # nbr_iter = 5
    # load_model = False
    # log = False

    if verbose:
        print(datetime.datetime.now(), 'Loading %s dataset' % dataset_name)

    logging.info('%s Loading %s dataset' % (datetime.datetime.now(), dataset_name))

    X, y, e, f = datamanager.get_dataset(dataset_name, datasets_path)
    features = f
    encoder = e

    if verbose:
        print(datetime.datetime.now(), 'Building preprocessing pipe')

    logging.info('%s Building preprocessing pipe' % datetime.datetime.now())

    preprocessing_pipe = prep.build_preprocessing_pipe()  #[0:5]

    if not load_model:
        nbr_models = nbr_splits * nbr_iter * len(preprocessing_pipe)
        if verbose:
            print(datetime.datetime.now(), 'Training and testing %d models' % nbr_models)

        logging.info('%s Training and testing %d models' % (datetime.datetime.now(), nbr_models))

        trained_model = stability.train_model(X, y, model_name, preprocessing_pipe, fit_predict_model, features,
                                              nbr_splits, nbr_iter, verbose, logging)

        if verbose:
            print(datetime.datetime.now(), 'Storing models')

        logging.info('%s Storing models' % datetime.datetime.now())

        stability.store_model(trained_model, model_name, dataset_name, models_path)
    else:
        if verbose:
            print(datetime.datetime.now(), 'Loading models')

        logging.info('%s Loading models' % datetime.datetime.now())

        trained_model = stability.load_model(model_name, dataset_name, models_path)

    if verbose:
        print(datetime.datetime.now(), 'Evaluating models stability')

    logging.info('%s Evaluating models stability' %datetime.datetime.now())
    model_stability = stability.evaluate_model_stability(model_name, trained_model,
                                                         analyze_model, aggregation_functions, encoder)

    if verbose:
        print(datetime.datetime.now(), 'Evaluating and comparing models stability')

    logging.info('%s Evaluating and comparing models stability' % datetime.datetime.now())
    model_stability_comparison = stability.evaluate_model_stability_comparison(model_name, trained_model,
                                                                               compare_models, aggregation_functions,
                                                                               encoder)

    if verbose:
        print('%s Storing results' % datetime.datetime.now())

    logging.info('%s Storing results' % datetime.datetime.now())
    stability.store_model_stability(model_stability, model_stability_comparison, aggregation_functions,
                                    model_name, dataset_name, results_path)

    if verbose:
        print(datetime.datetime.now(), 'Evaluation completed')

    logging.info('%s Evaluation completed' % datetime.datetime.now())


def main():

    parser = set_argparser()
    args = parser.parse_args()

    if args.sd:
        for d in sorted(datamanager.datasets):
            print(d)
        return 0

    if args.sm:
        for m in sorted(fit_predict_model):
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

    run_evaluate_stability(dataset_name, model_name, datasets_path, models_path, results_path,
                           nbr_splits, nbr_iter, load_model, verbose)


if __name__ == "__main__":
    main()
