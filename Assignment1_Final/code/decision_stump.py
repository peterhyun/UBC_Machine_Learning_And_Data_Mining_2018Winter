import numpy as np
import utils


class DecisionStumpEquality:

    def __init__(self):
        pass


    def fit(self, X, y):
        N, D = X.shape
        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)    
        #print(count) #for debugging, I added this #234, 166. 234 of 0s, 166 of 1s
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.splitSat = y_mode  # 0
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further (never the case cause both 0 and 1 appears)
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)  # minError is 166 

        # Loop over features looking for the best split
        X = np.round(X) #literally rounds all the data
        
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]
                #value is either longtitude or latitude

                # Find most likely class for each split
                
                #y_sat is either 0 or 1, and y[...] prints its value only if X[:,d] == value (y is a column vector with 0 or 1)
                y_sat = utils.mode(y[X[:,d] == value])
                #y_not is also either 0 or 1
                y_not = utils.mode(y[X[:,d] != value])

                # Make predictions
                y_pred = y_sat * np.ones(N) #either 400 size array of 0s or 1s
                y_pred[X[:, d] != value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):

        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] == self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat





class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y,minlength = 2)    
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count) 

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        # X = np.round(X)

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] > value])
                # y_not = The one that has more between 0 and 1 that's not the same with value
                y_not = utils.mode(y[X[:,d] <= value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] <= value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):

        M, D = X.shape
        # X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] > self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat



"""
A helper function that computes the entropy of the 
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to 1.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true (you can show this with l'Hopital's rule), 
but numerically results in NaN because log(0) returns -Inf.
"""
def entropy(p):
    plogp = 0*p # initialize full of zeros
    plogp[p>0] = p[p>0]*np.log(p[p>0]) # only do the computation when p>0
    return -np.sum(plogp)
    
# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?
class DecisionStumpInfoGain(DecisionStumpErrorRate):
    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y, minlength = 2)    
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count) 

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        maxIG = 0

        # Loop over features looking for the best split
        # X = np.round(X)

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                y_normalize = np.divide(y,float(y.size))
                # Find most likely class for each split
                y_yes = y[X[:,d] > value]
                # y_no = The one that has more between 0 and 1 that's not the same with value
                y_no = y[X[:,d] <= value]

                #print(y_yes)
                #print(y_no)
                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] > value])
                # y_not = The one that has more between 0 and 1 that's not the same with value
                y_not = utils.mode(y[X[:,d] <= value])

                y_yes_bincount = np.bincount(y_yes, minlength = 2)
                y_no_bincount = np.bincount(y_no, minlength = 2)

                # print(y_yes.size)
                y_yes_size = y_yes.size
                if (y_yes_size == 0):
                    y_yes_size = 1
            
                # print(y_no.size)
                y_no_size = y_no.size
                if (y_no_size == 0):
                    y_no_size = 1

                y_yes_bincount = np.divide(y_yes_bincount, float(y_yes_size))
                
                y_no_bincount = np.divide(y_no_bincount, float(y_no_size))

                # Compute IG
                IG = entropy(y_normalize) - (y_yes.size/float(N)*entropy(y_yes_bincount)) - (y_no.size/float(N)*entropy(y_no_bincount))

                # Compare to minimum error so far
                if IG > maxIG:
                    # This is the lowest error, store this value
                    maxIG = IG
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not


