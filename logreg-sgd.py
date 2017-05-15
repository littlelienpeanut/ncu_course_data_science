#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import math
from sympy import diff
import operator
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('HTRU_2.csv', header=None, names=['x%i' % (i) for i in xrange(8)] + ['y'])
    X = numpy.asarray(data[['x%i' % (i) for i in xrange(8)]])
    y = numpy.asarray(data['y'])

    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale


def cross_entropy(y, y_hat):
    loss = 0
    for i in xrange(len(y)):
        loss += -(y[i]*math.log(y_hat[i]) + (1-y[i])*math.log(1-y_hat[i]))
    return loss


def logreg_sgd(X, y, alpha = .001, iters = 100000, eps=1e-4):
    # TODO: compute theta
    n, d = X.shape
    theta = numpy.zeros((d, 1))
    y = y.reshape(8949,1)

    for i in range(0, iters):
        y_hat = predict_prob(X, theta)
        loss = y - y_hat
        gradient = numpy.dot(X.transpose(), loss) * -1
        theta = theta - alpha * gradient.reshape(8,1)
        print(i)

    return theta


def predict_prob(X, theta):
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))


def plot_roc_curve(y_test, y_prob):
    # TODO: compute tpr and fpr of different thresholds
    #variable
    y_test = y_test.tolist()
    y_prob = y_prob.tolist()
    tmp_y_prob = []
    tmp_y_prob = sorted(y_prob, key=float, reverse=True)
    count_1 = 0
    count_0 = 0
    TP = 0
    FP = 0
    tpr = []
    fpr = []
    sort_y_prob = []
    for i in range(0, len(y_prob), 1):
        sort_y_prob.append(y_test[y_prob.index(tmp_y_prob[i])])
    count_1 = sort_y_prob.count(1)
    count_0 = sort_y_prob.count(0)

    for i in range(len(sort_y_prob)):
        if sort_y_prob[i] == 1:
            TP += 1
            tpr.append(TP/count_1)
            fpr.append(FP/count_0)
        else :
            FP += 1
            tpr.append(TP/count_1)
            fpr.append(FP/count_0)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(fpr, tpr, 'b')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("roc_curve.png")
    plt.show()


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    print(theta)
    y_prob = predict_prob(X_train_scale, theta)
    print("Logreg train accuracy: %f" % (sklearn.metrics.accuracy_score(y_train, y_prob > .5)))
    y_prob = predict_prob(X_test_scale, theta)
    print("Logreg test accuracy: %f" % (sklearn.metrics.accuracy_score(y_test, y_prob > .5)))
    plot_roc_curve(y_test.flatten(), y_prob.flatten())


if __name__ == "__main__":
    main(sys.argv)
