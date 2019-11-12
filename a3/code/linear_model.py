import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        self.w = solve(X.T@z@X, X.T@z@y)

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        f = np.sum(np.log(np.exp(X@w - y)+np.exp(y - X@w)))
        print('f is '+str(f))
        # Calculate the gradient value
        g = np.array([np.sum(np.multiply((np.exp(X@w-y)-np.exp(y-X@w)),X)/(np.exp(X@w-y)+np.exp(y-X@w)))])
        print('g is '+str(g))

        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        M, N = X.shape
        Z = np.concatenate((np.ones(M)[:,np.newaxis], X), axis=1)
        self.V = solve(Z.T@Z,Z.T@y)

    def predict(self, X):
        M, N = X.shape
        Z = np.concatenate((np.ones(M)[:,np.newaxis], X), axis=1)
        return Z@self.V

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        ''' YOUR CODE HERE '''
        M, N = X.shape
        Z = np.ones(M)
        Z = np.reshape(Z, (M,1))
        for i in range(0,self.p):
            z = np.power(X,i+1)
            Z = np.concatenate((Z,z), axis = 1)
        return Z

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        Z_TtimesZ = Z.T@Z
        if(Z_TtimesZ.ndim == 0):
            Z_TtimesZ = np.reshape(Z_TtimesZ, (1,1))
        Z_Ttimesy = Z.T@y
        if(Z_Ttimesy.ndim == 0):
            Z_Ttimesy = np.reshape(Z_Ttimesy, (1,1))
        self.V = solve(Z_TtimesZ, Z_Ttimesy)

    def predict(self, X):
        Z = self.__polyBasis(X)
        return Z@self.V
