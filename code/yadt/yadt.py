
import os
import numpy as np
import csv
import subprocess
import pandas as pd
import shutil
from sklearn.base import BaseEstimator, ClassifierMixin
#import networkx as nx
#import pydotplus

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
        res.append( (col, col_type, feat_type ) )
    return res

def to_yadt(df, metadata, decision=None, targetname=None, 
             filebase='dataset', filenames=None, filedata=None, 
             sep=';', comp=False):
    if targetname is not None:
        target = targetname
    elif decision is not None and isinstance(decision, pd.DataFrame):
        target = decision.columns[0]
    else: 
        target = 'target'
    if filenames==None:
        filenames = filebase+'.names'
    if filedata==None:
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
        self.name = wdir+'_YaDT_{}'.format( id(self) )

    def fit(self, X=None, y=None, sample_weight=None, verbose=False, 
            filenames=None, filedata=None, deletefiles=False):
        if X is None:
            if y is not None:
                raise Exception("YaDTClassifier: no data but class given to fit")
            if filenames is None or filedata is None:
                raise Exception("YaDTClassifier: no data to fit")
        else:
            if filenames is not None or filedata is not None:
                raise Exception("YaDTClassifier: both data and file provided")
            if y is None:
                raise Exception("YaDTClassifier: data but no class given to fit")
            #print(X[:5])
            #if sample_weight is not None:
            #    print(sample_weight[:5])
            #print(X.shape)
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
                #m = min(sample_weight)
                #sample_weight *= ts.shape[0]
                w = self.name+'.weight'
                meta.append( (w, 'double', 'weights') )
                if isinstance(X, pd.DataFrame):
                    ts[w] = sample_weight
                else:
                    wcol = np.reshape(sample_weight, (sample_weight.shape[0], 1))
                    ts = np.column_stack((ts, wcol))
                    #print(ts[:5,:])
            to_yadt(df=ts, metadata=meta, decision=y, 
                     filenames=namesfile, filedata=datafile, 
                     sep=self.sep)
        cmd = 'dTcmd -fm {} -fd {} -sep {} -tb {} {}'.format(
                namesfile, datafile, self.sep, self.name+'.tree', self.options)
        if verbose:
            print(cmd)
        output = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT)
        if verbose:
            print(output)
    #    dt = nx.DiGraph(nx.drawing.nx_pydot.read_dot(name+'.dot'))
    #    dt_dot = pydotplus.graph_from_dot_data(open(name+'.dot', 'r').read())
        if deletefiles and os.path.exists(namesfile):
            os.remove(namesfile)
        if deletefiles and os.path.exists(datafile):
            os.remove(datafile)
        return self

    def predict(self, X=None, datafile=None, deletefiles=True, proba=False, verbose=False):
        if proba and (self.n_classes_ != 2):
            raise Exception("YaDT scoring does not currently support more than 2 classes")
        if X is None:
            if datafile is None:
                raise Exception("YaDTClassifier: no data to predict")
        else:
            if datafile is not None:
                raise Exception("YaDTClassifier: both df and datafile provided")
            datafile = self.name + ".test.gz"
            # remove weights column if present
            columns = [x[0] for x in self.metadata]
            df = X[columns] if isinstance(X, pd.DataFrame) else X
            to_yadt_data(df, datafile, sep=self.sep)
        scorefile = self.name + ".score"
        cmd = 'dTcmd -bt {} -fs {} -sep {} -s {}'.format(
            self.name+'.tree', datafile, self.sep, scorefile)
        if verbose:
            print(cmd)
        output = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT)
        if verbose:
            print(output)
        res = pd.read_csv(scorefile, sep=self.sep, header=None, usecols=(None if proba else (0,)))
        if deletefiles and os.path.exists(datafile):
            os.remove(datafile)
        if os.path.exists(scorefile):
            os.remove(scorefile)
        if not proba:
            return res[0].values
        r0 = res[0].values
        r1 = res[1].values
        base_class = self.classes_[0]
        r = (r0==base_class)*r1 + (r0!=base_class)*(1-r1)
        r = r.astype(np.float64)
        #print(r0)
        #print(base_class)
        #print(r)
        return np.column_stack( (r, 1-r) )
    
    def predict_proba(self, X=None, datafile=None, deletefiles=True, verbose=False):
        return self.predict(X, datafile, deletefiles, proba=True, verbose=verbose)
    
    def decision_function(self, X=None, datafile=None, deletefiles=True, verbose=False):
        proba = self.predict(X, datafile, deletefiles, proba=True, verbose=verbose)
        return [ 1-2*x[0] for x in proba ]
