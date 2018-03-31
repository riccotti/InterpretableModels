__author__ = "Riccardo Guidotti"

import numpy as np
import networkx as nx

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from scipy.stats import kendalltau
from scipy.stats import spearmanr

from scipy.spatial.distance import cityblock
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine

from yadt import yadt
from apted import apted, helpers

from jrbc.rule_based_classifier import RuleBasedClassifier

from scipy.stats import entropy
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")


linear_models = {
    'LinearRegression': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso,
}

decision_trees = {
    'DecisionTreeClassifier': DecisionTreeClassifier,
}


def sample_pearson(s1, s2, nf):
    # nf = len(s1 | s2)  # the total number of features
    ki = len(s1)  # number of features in s1
    kj = len(s2)  # number of features in s2
    den = np.sqrt(ki * (nf - ki)) * np.sqrt(kj * (nf - kj))
    if den == 0:
        d = 0.0 if (ki == 0 and kj != 0) or (ki != 0 and kj == 0) \
                   or (ki == nf and kj != nf) or (ki != nf and kj == nf) else 1.0
    else:
        rij = len(s1 & s2)
        d = (nf * rij - ki * kj) / den
    return d


def intersection(fset1, fset2):
    return len(fset1 & fset2)


def jaccard(fset1, fset2):
    if len(fset1) == 0 and len(fset2) == 0:
        return 0.0
    return len(fset1 & fset2) / len(fset1 | fset2)


def fit_predict_sklearn_decision_tree(train_test, seed, features):
    clf = DecisionTreeClassifier(random_state=seed)
    X_train, X_test, y_train, y_test, fsindexes = train_test
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return clf, acc, f1, fsindexes


def fit_predict_yadt_decision_tree(train_test, seed, features):
    X_train, X_test, y_train, y_test, fsindexes = train_test
    metadata = [f for fs, f in zip(fsindexes, features) if fs]
    clf = yadt.YaDTClassifier(metadata, options='-m 2')
    clf.fit(X_train, y_train, verbose=False, deletefiles=True)
    y_pred = clf.predict(X_test, verbose=False, deletefiles=True)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    # yadt.clean()
    return clf, acc, f1, fsindexes


def count_leaves(tree, n_nodes):
    children_left = tree.children_left
    children_right = tree.children_right

    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    return np.sum(is_leaves)


def get_tree_structure(tree):
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    return node_depth, is_leaves, children_left, children_right


def get_max_width(node_depth):
    return np.max(np.unique(node_depth, return_counts=True)[1])


def get_brackets(node_id, children_left, children_right, is_leaves, mid, brackets_repository):
    if mid in brackets_repository:
        return brackets_repository[mid]

    if is_leaves[node_id]:
        return node_id
    brackets = '%s' % node_id
    if children_left[node_id] >= 0:
        brackets += '{%s}' % get_brackets(children_left[node_id], children_left, children_right, is_leaves,
                                          mid, brackets_repository)
    if children_right[node_id] >= 0:
        brackets += '{%s}' % get_brackets(children_right[node_id], children_left, children_right, is_leaves,
                                          mid, brackets_repository)
    if node_id != 0:
        return brackets
    else:
        brackets_repository[mid] = helpers.Tree.from_text('{%s}' % brackets)
        return brackets_repository[mid]


def get_balancing(node_id, children_left, children_right, is_leaves):
    if is_leaves[node_id]:
        return 0
    hl = 0
    hr = 0
    if children_left[node_id] >= 0:
        hl = get_balancing(children_left[node_id], children_left, children_right, is_leaves)
    if children_right[node_id] >= 0:
        hr = get_balancing(children_right[node_id], children_left, children_right, is_leaves)
    h = max(hr, hl) + 1
    if node_id != 0:
        return h
    else:
        return hr - hl


def get_features_set(mfi, e, f, cond):

    if e is None:
        fset = set([fid for fid, fimp in enumerate(mfi) if cond(fimp)])
    else:
        fset = set()
        idx = 0
        idx_m = 0
        for i in range(0, len(e.feature_indices_) - 1):
            fimp = 0
            for j in range(e.feature_indices_[i], e.feature_indices_[i + 1]):
                if f[j]:
                    fimp += mfi[idx_m]
                    idx_m += 1
            if cond(fimp):
                fset.add(idx)
                idx += 1
        for i in range(idx_m, len(mfi)):
            fimp = mfi[i]
            if cond(fimp):
                fset.add(idx)
                idx += 1

    return fset


def get_features_rank(mfi, e, f):

    if e is None:
        frank = mfi
    else:
        frank = list()
        idx_m = 0
        for i in range(0, len(e.feature_indices_) - 1):
            fimp = 0
            for j in range(e.feature_indices_[i], e.feature_indices_[i + 1]):
                if f[j]:
                    fimp += mfi[idx_m]
                    idx_m += 1
            frank.append(fimp)
        for i in range(idx_m, len(mfi)):
            fimp = mfi[i]
            frank.append(fimp)

    return frank


def analyze_sklearn_decision_tree(m, e, f):
    fset = get_features_set(m.feature_importances_, e, f, lambda x: x > 0.0)

    nbr_features = len(fset)
    node_depth, is_leaves, children_left, children_right = get_tree_structure(m.tree_)
    max_depth = m.tree_.max_depth
    max_width = get_max_width(node_depth)
    nbr_nodes = m.tree_.node_count
    nbr_leaves = count_leaves(m.tree_, m.tree_.node_count)
    nbr_splits = m.tree_.node_count - nbr_leaves
    balancing = get_balancing(0, children_left, children_right, is_leaves)

    meval = {
        'nbr_features': nbr_features,
        'max_depth': max_depth,
        'max_width': max_width,
        'nbr_nodes': nbr_nodes,
        'nbr_leaves': nbr_leaves,
        'nbr_splits': nbr_splits,
        'balancing': balancing,
    }

    return meval


def get_node_labels(dt):
    return {k: v.replace('"', '').replace('\\n', '') for k, v in nx.get_node_attributes(dt, 'label').items()}


def get_edge_labels(dt):
    return {k: v.replace('"', '').replace('\\n', '') for k, v in nx.get_edge_attributes(dt, 'label').items()}


def analyze_yadt_decision_tree(m, e, f):
    dt = m.dt
    # edge_labels = get_edge_labels(dt)
    node_labels = get_node_labels(dt)
    node_isleaf = {k: v == 'ellipse' for k, v in nx.get_node_attributes(dt, 'shape').items()}

    fset = set([v for k, v in node_labels.items() if not node_isleaf[k]])
    nbr_features = len(fset)
    nbr_nodes = dt.number_of_nodes()
    nbr_leaves = np.sum([1 for v in node_isleaf.values() if v])
    nbr_splits = nbr_nodes - nbr_leaves

    node_depth = dict()
    stack = [('n0', -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        for k in dt[node_id]:
            stack.append((k, parent_depth + 1))

    max_depth = np.max(list(node_depth.values()))
    max_width = np.max(np.unique(list(node_depth.values()), return_counts=True)[1])

    meval = {
        'nbr_features': nbr_features,
        'max_depth': max_depth,
        'max_width': max_width,
        'nbr_nodes': nbr_nodes,
        'nbr_leaves': nbr_leaves,
        'nbr_splits': nbr_splits,
        # 'balancing': balancing, yadt is not binary thus is not available
    }

    return meval


# def java_edit_distance(tb1, tb2):
#     cmd = 'java -jar apted.jar -t %s %s' % (tb1, tb2)
#     ed = float(subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT))
#     return ed

def tree_edit_distance(t1, t2):
    apted_run = apted.APTED(t1, t2)
    ted = apted_run.compute_edit_distance()
    return ted


def compare_sklearn_decision_trees(m1, m2, e, f1, f2, args):
    # f1set = set([fid for fid, fimp in enumerate(m1.feature_importances_) if fimp > 0.0])
    # f2set = set([fid for fid, fimp in enumerate(m2.feature_importances_) if fimp > 0.0])
    # f1rank = m1.feature_importances_
    # f2rank = m2.feature_importances_

    f1set = get_features_set(m1.feature_importances_, e, f1, lambda x: x > 0.0)
    f2set = get_features_set(m2.feature_importances_, e, f2, lambda x: x > 0.0)
    f1rank = get_features_rank(m1.feature_importances_, e, f1)
    f2rank = get_features_rank(m2.feature_importances_, e, f2)
    nf = len([f for f in f1 if f])

    if len(f1rank) < len(f2rank):
        f1rank = np.concatenate((f1rank, [0.0] * (len(f2rank) - len(f1rank))))
    elif len(f2rank) < len(f1rank):
        f2rank = np.concatenate((f2rank, [0.0] * (len(f1rank) - len(f2rank))))

    # node_depth1, is_leaves1, cleft1, cright1 = get_tree_structure(m1.tree_)
    # node_depth2, is_leaves2, cleft2, cright2 = get_tree_structure(m2.tree_)
    #
    # mid1, mid2, brackets_repository = args
    # tree_brackets1 = get_brackets(0, cleft1, cright1, is_leaves1, mid1, brackets_repository)
    # tree_brackets2 = get_brackets(0, cleft2, cright2, is_leaves2, mid2, brackets_repository)
    #
    # ted = tree_edit_distance(tree_brackets1, tree_brackets2)

    meval = {
        'intersection': intersection(f1set, f2set),
        'jaccard': jaccard(f1set, f2set),
        'sample_pearson': sample_pearson(f1set, f2set, nf),
        'kendalltau': rank_kendalltau(f1rank, f2rank),
        'spearmanr': rank_spearmanr(f1rank, f2rank),
        # 'tree_edit_distance': ted,
    }

    return meval


def compare_yadt_decision_trees(m1, m2, e, f1, f2, args):
    dt1 = m1.dt
    node_labels1 = get_node_labels(dt1)
    node_isleaf1 = {k: v == 'ellipse' for k, v in nx.get_node_attributes(dt1, 'shape').items()}
    f1set = set([v for k, v in node_labels1.items() if not node_isleaf1[k]])

    dt2 = m2.dt
    node_labels2 = get_node_labels(dt2)
    node_isleaf2 = {k: v == 'ellipse' for k, v in nx.get_node_attributes(dt2, 'shape').items()}
    f2set = set([v for k, v in node_labels2.items() if not node_isleaf2[k]])

    nf = len([f for f in f1 if f])

    meval = {
        'intersection': intersection(f1set, f2set),
        'jaccard': jaccard(f1set, f2set),
        'sample_pearson': sample_pearson(f1set, f2set, nf),
        # 'kendalltau': rank_kendalltau(f1rank, f2rank), # feature importance rank not available for yadt
        # 'spearmanr': rank_spearmanr(f1rank, f2rank),
    }

    return meval


def fit_predict_linear_regression(train_test, seed, features):
    clf = LinearRegression()
    X_train, X_test, y_train, y_test, fsindexes = train_test
    clf.fit(X_train, y_train)
    y_pred = np.round(clf.predict(X_test)).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return clf, acc, f1, fsindexes


def fit_predict_lasso(train_test, seed, features):
    clf = Lasso(random_state=seed)
    X_train, X_test, y_train, y_test, fsindexes = train_test
    clf.fit(X_train, y_train)
    y_pred = np.round(clf.predict(X_test)).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return clf, acc, f1, fsindexes


def fit_predict_ridge(train_test, seed, features):
    clf = Ridge(random_state=seed)
    X_train, X_test, y_train, y_test, fsindexes = train_test
    clf.fit(X_train, y_train)
    y_pred = np.round(clf.predict(X_test)).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return clf, acc, f1, fsindexes


def analyze_sklearn_linear_models(m, e, f):
    fset = get_features_set(m.coef_, e, f, lambda x: x != 0.0)
    fpset = get_features_set(m.coef_, e, f, lambda x: x > 0.0)
    fnset = get_features_set(m.coef_, e, f, lambda x: x < 0.0)

    meval = {
        'nbr_features': len(fset),
        'nbr_pos_features': len(fpset),
        'nbr_neg_features': len(fnset),
    }

    return meval


def normalized_euclidean_distance(x, y):
    if len(x) == 1:
        return 0.0
    if np.sum(x) == 0 and np.sum(y) == 0:
        return 1.0
    if len(np.unique(x)) == 1 and len(np.unique(y)) == 1:
        den = max(x[0], y[0])
        num = min(x[0], y[0])
        return num / den
    return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))


def rank_kendalltau(x, y):
    if np.sum(x) == 0 and np.sum(y) == 0:
        return 1.0
    if len(x) < 3:
        if len(x) == 1:
            den = max(x[0], y[0])
            num = min(x[0], y[0])
        else:
            den = max(np.abs(x[0] - x[1]), np.abs(y[0] - y[1]))
            num = min(np.abs(x[0] - x[1]), np.abs(y[0] - y[1]))
        return num / den
    if len(np.unique(x)) == 1 and len(np.unique(y)) == 1:
        den = max(x[0], y[0])
        num = min(x[0], y[0])
        return num / den
    if len(np.unique(x)) == 1 and len(np.unique(y)) == 1:
        return 0.0
    return kendalltau(x, y)[0]


def rank_spearmanr(x, y):
    if np.sum(x) == 0 and np.sum(y) == 0:
        return 1.0
    if len(x) < 3:
        if len(x) == 1:
            den = max(x[0], y[0])
            num = min(x[0], y[0])
        else:
            den = max(np.abs(x[0] - x[1]), np.abs(y[0] - y[1]))
            num = min(np.abs(x[0] - x[1]), np.abs(y[0] - y[1]))
        return num / den
    if len(np.unique(x)) == 1 and len(np.unique(y)) == 1:
        den = max(x[0], y[0])
        num = min(x[0], y[0])
        return num / den
    if len(np.unique(x)) == 1 and len(np.unique(y)) == 1:
        return 0.0
    return spearmanr(x, y)[0]


def imcosine(x, y):
    if np.sum(x) == 0 and np.sum(y) == 0:
        return 1.0
    return cosine(x, y)


def get_weights(m, f):
    idx = 0
    mc = list()
    for i in range(0, len(f)):
        if f[i]:
            mc.append(m[idx])
            idx += 1
        else:
            mc.append(0)
    return np.array(mc)


def compare_sklearn_linear_models(m1, m2, e, f1, f2, args):
    # f1set = set([fid for fid, fimp in enumerate(m1.coef_) if fimp != 0.0])
    # f2set = set([fid for fid, fimp in enumerate(m2.coef_) if fimp != 0.0])
    # f1rank = m1.coef_
    # f2rank = m2.coef_

    f1set = get_features_set(m1.coef_, e, f1, lambda x: x != 0.0)
    f2set = get_features_set(m2.coef_, e, f2, lambda x: x != 0.0)

    if len(f1set) == 0:
        f1set = set([fid for fid, fimp in enumerate(m1.coef_)])
    if len(f2set) == 0:
        f2set = set([fid for fid, fimp in enumerate(m2.coef_)])

    f1rank = get_features_rank(m1.coef_, e, f1)
    f2rank = get_features_rank(m2.coef_, e, f2)

    if len(f1rank) < len(f2rank):
        f1rank = np.concatenate((f1rank, [0.0] * (len(f2rank) - len(f1rank))))
    elif len(f2rank) < len(f1rank):
        f2rank = np.concatenate((f2rank, [0.0] * (len(f1rank) - len(f2rank))))

    mc1 = get_weights(m1.coef_, f1)
    mc2 = get_weights(m2.coef_, f2)

    nf = len([f for f in f1 if f])

    meval = {
        'intersection': intersection(f1set, f2set),
        'jaccard': jaccard(f1set, f2set),
        'sample_pearson': sample_pearson(f1set, f2set, nf),
        'kendalltau': rank_kendalltau(f1rank, f2rank),
        'spearmanr': rank_spearmanr(f1rank, f2rank),
        'euclidean': euclidean(mc1, mc2),
        'neuclidean': normalized_euclidean_distance(mc1, mc2),
        'cosine': imcosine(mc1, mc2),
        'cityblock': cityblock(mc1, mc2),
    }

    return meval


def fit_predict_cpar(train_test, seed, features):
    rbc = RuleBasedClassifier(alg='CPAR', options='-Xmx4G')
    X_train, X_test, y_train, y_test, fsindexes = train_test
    rbc.fit(X_train, y_train, verbose=False, deletefile=True)
    y_pred = rbc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return rbc, acc, f1, fsindexes


def fit_predict_foil(train_test, seed, features):
    rbc = RuleBasedClassifier(alg='FOIL', options='-Xmx4G')
    X_train, X_test, y_train, y_test, fsindexes = train_test
    rbc.fit(X_train, y_train, verbose=False, deletefile=True)
    y_pred = rbc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return rbc, acc, f1, fsindexes


def get_feature_set_rules(rules):
    fset = set()
    rule_len = list()
    prob = defaultdict(int)
    for rule in rules:
        cons, ant, laplace = rule
        for col, val in ant.items():
            fset.add(col)
        rule_len.append(len(ant))
        prob[cons] += 1

    avg_rule_len = np.mean(rule_len)
    rule_entropy = entropy([c/len(rules) for c in prob.values()])
    return fset, avg_rule_len, rule_entropy


def get_feature_rank_rules(rules, nf):
    frank = [0.0] * nf
    for rule in rules:
        cons, ant, laplace = rule
        for col, val in ant.items():
            if frank[col] == 0.0:
                frank[col] = laplace

    return frank


def analyze_rule_based(m, e, f):
    fset, avg_rule_len, rule_entropy = get_feature_set_rules(m.rules)

    meval = {
        'nbr_features': len(fset),
        'nbr_rules': len(m.rules),
        'avg_rule_len': avg_rule_len,
        'rule_entropy': rule_entropy
    }

    return meval


def compare_rule_based(m1, m2, e, f1, f2, args):
    f1set, _, _ = get_feature_set_rules(m1.rules)
    f2set, _, _ = get_feature_set_rules(m2.rules)

    nf1 = len([f for f in f1 if f])
    nf2 = len([f for f in f2 if f])
    nf = max(nf1, nf2)

    f1rank = get_feature_rank_rules(m1.rules, nf)
    f2rank = get_feature_rank_rules(m2.rules, nf)

    meval = {
        'intersection': intersection(f1set, f2set),
        'jaccard': jaccard(f1set, f2set),
        'sample_pearson': sample_pearson(f1set, f2set, nf),
        'kendalltau': rank_kendalltau(f1rank, f2rank),
        'spearmanr': rank_spearmanr(f1rank, f2rank),
    }

    return meval

