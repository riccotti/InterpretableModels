
import os
import csv
import shutil
import subprocess
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.base import BaseEstimator, ClassifierMixin

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def metadata(df, ovr_types={}):
    res = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == np.float64:
            col_type = 'double'
            feat_type = 'continuous'
        else:
            kind = dtype.kind
            if kind == 'f':
                col_type = 'float'
                feat_type = 'continuous'
            elif kind == 'i':
                col_type = 'integer'
                feat_type = 'continuous'
            else:
                col_type = 'string'        
                feat_type = 'discrete'
        if col in ovr_types:
            feat_type = ovr_types[col]
        res.append((col, col_type, feat_type))
    return res


def to_yadt(df, metadata, decision=None, targetname=None, 
             filebase='dataset', filenames=None, filedata=None, 
             sep=';', comp=False):
    if targetname is not None:
        target = targetname
    elif decision is not None and isinstance(decision, pd.DataFrame):
        target = decision.columns[0]
    else: 
        target = 'class'
    if filenames is None:
        filenames = filebase+'.names'
    if filedata is None:
        filedata = filebase+('.data.gz' if comp else '.data')
    columns = [x[0] for x in metadata]
    if decision is None:
        meta = metadata
        alldata = df
    else:
        meta = metadata + [(target, 'string', 'class')]
        if isinstance(df, pd.DataFrame):
            alldata = df[columns]
        else:
            alldata = pd.DataFrame(df)
        alldata[target] = decision
    to_yadt_names(meta, filenames, sep=sep)
    to_yadt_data(alldata, filedata, sep=sep)


def to_yadt_names(metadata, filename, sep=';'):
    with open(filename, 'w') as names_file:
        for col in metadata:
            print("{}{}{}{}{}".format(col[0], sep, col[1], sep, col[2]), file=names_file)


def to_yadt_data(df, filename, sep=';'):
    comp = 'gzip' if filename.endswith('.gz') else None
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df) #TBD: float format becomes longer
    df.to_csv(filename, na_rep='?', quoting=csv.QUOTE_NONE, compression=comp,
              sep=sep, index=False, float_format='%g', header=False)


wdir = "__yadtcache__/"


def clean():
    global wdir
    if os.path.exists(wdir):
         shutil.rmtree(wdir)


class YaDTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, metadata, sep=';', options=""):
        global wdir
        if not os.path.exists(wdir):
            os.mkdir(wdir)
        self.metadata = metadata
        self.sep = sep
        self.options = options
        self.name = wdir+'_YaDT_{}'.format(id(self))
        self.dt = None

    def fit(self, X=None, y=None, sample_weight=None, verbose=False, 
            filenames=None, filedata=None, deletefiles=False):
        if X is None:
            if y is not None:
                raise Exception("YaDTClassifier: no data but class given to fit")
            if filenames is None or filedata is None:
                raise Exception("YaDTClassifier: no data to fit")

        if filenames is not None or filedata is not None:
            raise Exception("YaDTClassifier: both data and file provided")
        if y is None:
            raise Exception("YaDTClassifier: data but no class given to fit")

        if isinstance(X, pd.DataFrame):
            columns = [x[0] for x in self.metadata]
            ts = X[columns]
        else:
            ts = X

        self.classes_ = np.array(y.unique() if isinstance(y, pd.DataFrame) else np.unique(y))
        self.n_classes_ = self.classes_.shape[0]
        namesfile = self.name + ".names"
        datafile = self.name + ".data.gz"
        meta = self.metadata.copy()

        if sample_weight is not None:
            w = self.name+'.weight'
            meta.append((w, 'double', 'weights'))
            if isinstance(X, pd.DataFrame):
                ts[w] = sample_weight
            else:
                wcol = np.reshape(sample_weight, (sample_weight.shape[0], 1))
                ts = np.column_stack((ts, wcol))
        to_yadt(df=ts, metadata=meta, decision=y, filenames=namesfile, filedata=datafile, sep=self.sep)

        cmd = "./yadt/dTcmd -fm {} -fd {} -sep '{}' -tb {} -d {} {}".format(
                namesfile, datafile, self.sep, self.name+'.tree', self.name+'.dot', self.options)
        if verbose:
            print(cmd)
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        if verbose:
            print(output)
        # print(self.name+'.dot')
        self.dt = nx.DiGraph(nx.drawing.nx_pydot.read_dot(self.name+'.dot'))
    #    dt_dot = pydotplus.graph_from_dot_data(open(name+'.dot', 'r').read())

        if deletefiles and os.path.exists(namesfile):
            os.remove(namesfile)
        if deletefiles and os.path.exists(datafile):
            os.remove(datafile)
        if os.path.exists(self.name+'.dot'):
            os.remove(self.name+'.dot')
        return self

    def predict(self, X=None, targetname='class', datafile=None, deletefiles=True, verbose=False):
        if X is None:
            if datafile is None:
                raise Exception("YaDTClassifier: no data to predict")
        else:
            if datafile is not None:
                raise Exception("YaDTClassifier: both df and datafile provided")
            datafile = self.name + ".test.gz"
            # remove weights column if present
            columns = [x[0] for x in self.metadata]
            if isinstance(X, pd.DataFrame):
                df = X[columns]
                df[targetname] = 0
            else:
                df = np.column_stack((X, [0] * len(X)))
            to_yadt_data(df, datafile, sep=self.sep)

        scorefile = self.name + ".score"
        cmd = "./yadt/dTcmd -bt {} -ft {} -sep '{}' -s {}".format(
            self.name+'.tree', datafile, self.sep, scorefile)
        if verbose:
            print(cmd)
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        if verbose:
            print(output)
        res = pd.read_csv(scorefile, sep=self.sep, header=None, usecols=None)
        if deletefiles and os.path.exists(datafile):
            os.remove(datafile)
        if deletefiles and os.path.exists(scorefile):
            os.remove(scorefile)
        if os.path.exists(self.name+'.tree'):
            os.remove(self.name+'.tree')
        return res[1].values

    def get_node_labels(self):
        return {k: v.replace('"', '').replace('\\n', '') for k, v in nx.get_node_attributes(self.dt, 'label').items()}

    def get_edge_labels(self):
        return {k: v.replace('"', '').replace('\\n', '') for k, v in nx.get_edge_attributes(self.dt, 'label').items()}

    def ypredict(self, X=None, targetname='class', features=None, default_value=None):
        class_name = targetname
        features_type = dict()
        discrete = list()
        continuous = list()
        features_order = dict()
        for i, f in enumerate(features):
            features_type[f[0]] = f[1]
            features_order[f[0]] = i
            if f[2] == 'continuous':
                continuous.append(f[0])
            elif f[2] == 'discrete':
                discrete.append(f[0])
        # features_type['class'] = 'string'
        return self.predict_nx(X, class_name, features_type, discrete, continuous, features_order, default_value,
                               leafnode=False)

    def predict_nx(self, X, class_name, features_type, discrete, continuous, features_order, default_value,
                   leafnode=True):
        edge_labels = self.get_edge_labels()
        node_labels = self.get_node_labels()
        node_isleaf = {k: v == 'ellipse' for k, v in nx.get_node_attributes(self.dt, 'shape').items()}

        # print(edge_labels)
        # print(node_labels)
        # print(node_isleaf)

        y_list = list()
        lf_list = list()
        for x in X:
            y, tp = self.predict_single_record(x, class_name, edge_labels, node_labels, node_isleaf,
                                               features_type, discrete, continuous, features_order, default_value)
            y_list.append(y)
            if tp is None:
                continue
            lf_list.append(tp[-1])

        if leafnode:
            return np.array(y_list), lf_list

        return np.array(y_list)

    def predict_single_record(self, x, class_name, edge_labels, node_labels, node_isleaf, features_type, discrete,
                              continuous, features_order, default_value, n_iter=1000):
        root = 'n0'
        node = root
        tree_path = list()
        count = 0
        while not node_isleaf[node]:
            att = node_labels[node]
            val = x[features_order[att]]
            for child in self.dt.neighbors(node):
                count += 1
                edge_val = edge_labels[(node, child)]
                if att in discrete:
                    val = val.strip() if isinstance(val, str) else val
                    if yadt_value2type(edge_val, att, features_type) == val:
                        tree_path.append(node)
                        node = child
                        break
                else:
                    pyval = yadt_value2type(val, att, features_type)
                    if '>' in edge_val:
                        thr = yadt_value2type(edge_val.replace('>', ''), att, features_type)
                        if pyval > thr:
                            tree_path.append(node)
                            node = child
                            break
                    elif '<=' in edge_val:
                        thr = yadt_value2type(edge_val.replace('<=', ''), att, features_type)
                        if pyval <= thr:
                            tree_path.append(node)
                            node = child
                            break
            if count >= n_iter:
                # print('Loop in Yadt prediction')
                return default_value, None
            count += 1
            # break

        tree_path.append(node)

        outcome = int(node_labels[node].split('(')[0])
        # print(outcome)
        # return -1
        # outcome = yadt_value2type(outcome, class_name, features_type)

        return outcome, tree_path


def yadt_value2type(x, attribute, features_type):

    if features_type[attribute] == 'integer':
        x = int(float(x))
    elif features_type[attribute] == 'double':
        x = float(x)

    return x

