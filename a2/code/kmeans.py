import numpy as np
from utils import euclidean_dist_squared

class Kmeans:

    def __init__(self, k):
        self.k = k


    def error(self, X):
        N, D = X.shape
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        argminIndex = np.argmin(dist2, axis=1)
        temp = np.zeros((N, 1))
        for i in range(N):
            temp[i] = dist2[i,argminIndex[i]]
            #print(temp[i])
            #print(temp.sum())   #biggest mistake was to do temp[i].sum()
        #return temp[i].sum()
        return np.sum(np.min(dist2,axis=1))

    def returnMeans(self):
        return self.means

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]

        self_error = 0
        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)
            # Update means
            for kk in range(self.k):
                if np.any(y==kk): # don't update the mean if no examples are assigned to it (one of several possible approaches)
                    means[kk] = X[y==kk].mean(axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))
            #Moved this hear.
            # Stop if no point changed cluster

            self.means = means
            print('Error is ')
            self_error = self.error(X)
            print(self_error)

            if changes == 0:
                break
        
        
            
        return self_error

    def predict(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)
        
