
import sys
import yadt
import pandas as pd
import scikitplot as skplt # pip install scikit-plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn import metrics
from sklearn import datasets
from sklearn.metrics import *


def model_task(model):
    global X_train, y_train, X_test, y_test
    print('-- {} --'.format(type(model)))

    # y_pred = model.fit_predict(X_train, y_train, X_test, verbose=True)
    # print(y_pred)

    """ model fitting """
    model.fit(X_train, y_train, verbose=True)
    print('Classes: {}'.format(model.classes_))

    """ model prediction """
    y_pred = model.predict(X_test, verbose=True)
    print(y_pred)
    # print("acc = {}".format(metrics.accuracy_score(y_test, y_pred)))
    # print(metrics.classification_report(y_test, y_pred))
    #
    # """ quality measures plots """
    # y_proba = model.predict_proba(X_test)
    # #print(type(y_proba))
    # #print(y_proba)
    # skplt.metrics.plot_precision_recall_curve(y_test, y_proba)
    # plt.show()
    # return y_proba

#dftrain = pd.read_pickle('dftrain.p.gz')

""" iris dataset from sklearn """
dataset = datasets.load_breast_cancer()
dataset = datasets.load_iris()
""" import from pickle format """
#_dataset = pd.read_pickle('breastcancer.p.gz')

features = dataset.feature_names
predictive = pd.DataFrame(dataset.data, columns=features)
decision = dataset.target

""" YaDT metadata """
metadata = yadt.metadata(predictive, ovr_types={})
""" export in YaDT format """
yadt.to_yadt(predictive, metadata, decision=decision,
              filebase='iris', targetname='class', comp=True)


X_train, X_test, y_train, y_test = train_test_split(
        predictive, decision, test_size=0.33, stratify=decision)


clf = yadt.YaDTClassifier(metadata, options='-m 2')
clf.fit(X_train, y_train, verbose=True)
y_pred = clf.predict(X_test, verbose=True, deletefiles=False)
print(y_pred)


# """ export in pickle format """
# #dataset.to_pickle('breastcancer.p.gz')
#
# """ import from YaDT format """
# # TBD
#
# """ train-test split """

#
# # model1 = DecisionTreeClassifier() #yadt.YaDTClassifier(X_train, options='-m 2 -npp')
# # y_proba1 = model_task(model1)
#
# model2 = yadt.YaDTClassifier(metadata, options='-m 2')
# y_proba2 = model_task(model2)
#
# # model3 = AdaBoostClassifier(
# #     base_estimator=yadt.YaDTClassifier(metadata, options='-m 2'),
# #     learning_rate=1,
# #     n_estimators=16,
# #     algorithm="SAMME.R")
# # y_proba3 = model_task(model3)
# #
# # model4 = RandomForestClassifier(n_estimators=32)
# # y_proba4 = model_task(model4)
# #
# #
# # # calibration comparison (for binary classifiers)
# # if model1.n_classes_ == 2:
# #     skplt.metrics.plot_calibration_curve(y_test, [y_proba1, y_proba2, y_proba3, y_proba4],
# #                                          [type(model1), type(model2), type(model3), type(model4)])
# #     plt.show()
#
# #yadt.clean()
#
#
