import numpy as np

class NaiveBayes:
    # Naive Bayes implementation.
    # Assumes the feature are binary.
    # Also assumes the labels go from 0,1,...C-1

    def __init__(self, num_classes, beta=0):
        self.num_classes = num_classes
        self.beta = beta

    def fit(self, X, y):
        N, D = X.shape
        #Rows: 8121, Columns: 100

        # Compute the number of class labels
        C = self.num_classes

        # Compute the probability of each class i.e p(y==c)
        counts = np.bincount(y)
        p_y = counts / N
        print(p_y) #1 * 4 Matrix

        # Compute the conditional probabilities i.e.
        # p(x(i,j)=1 | y(i)==c) as p_xy
        # p(x(i,j)=0 | y(i)==c) as p_xy
        p_xy = np.zeros((D, C))
        y_class = 0
        for d in range(D):
            count0 = 0
            count1 = 0
            count2 = 0
            count3 = 0
            for n in range(N):
                y_class = y[n]
                if (X[n, d] != 0) and (y_class == 0):
                    count0 += 1
                if (X[n, d] != 0) and (y_class == 1):
                    count1 += 1
                if (X[n, d] != 0) and (y_class == 2):
                    count2 += 1
                if (X[n, d] != 0) and (y_class == 3):
                    count3 += 1
            p_xy[d,0] = (count0/float(counts[0]))
            p_xy[d,1] = (count1/float(counts[1]))
            p_xy[d,2] = (count2/float(counts[2]))
            p_xy[d,3] = (count3/float(counts[3]))
        
        # TODO: replace the above line with the proper code 

        self.p_y = p_y
        self.p_xy = p_xy

    def predict(self, X):

        N, D = X.shape
        C = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(N)
        for n in range(N):

            probs = p_y.copy() # initialize with the p(y) terms
            for d in range(D):
                if X[n, d] != 0:
                    probs *= p_xy[d, :]
                else:
                    probs *= (1-p_xy[d, :])

            y_pred[n] = np.argmax(probs)

        return y_pred
