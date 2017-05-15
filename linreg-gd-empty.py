#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import numpy as np
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./ENB2012_data.csv')
    feature_col = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    label_col = ['Y1']
    X = data[feature_col].as_matrix()
    y = data[label_col].as_matrix()
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

def gradient_descent(X, y, alpha = .001, iters = 100000, eps=1e-4):
    # TODO: fill this procedure as an exercise
    n, d = X.shape
    theta = numpy.matrix(numpy.zeros((d, 1)))
    x_t = X.transpose()
    for i in range(0, iters):
        y_hat = np.dot(X, theta)
        loss = y_hat - y
        gradient = np.dot(x_t, loss)
        theta = theta - alpha * gradient

    return theta

def predict(X, theta):
    return numpy.dot(X, theta)

def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = gradient_descent(X_train_scale, y_train)
    y_hat = predict(X_train_scale, theta)


    print("Linear train R^2: %f" % (sklearn.metrics.r2_score(y_train, y_hat)))
    y_hat = predict(X_test_scale, theta)
    print("Linear test R^2: %f" % (sklearn.metrics.r2_score(y_test, y_hat)))


if __name__ == "__main__":
    main(sys.argv)
