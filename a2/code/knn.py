"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        D = utils.euclidean_dist_squared(self.X, Xtest)
        M, N = D.shape
        copy = D.copy()
        O, P = Xtest.shape
        y_pred = np.zeros(O)
        for i in range(N):
            temp = D[:,i].argsort()
            y_pred[i] = utils.mode(self.y[temp[0:self.k]])
        return y_pred