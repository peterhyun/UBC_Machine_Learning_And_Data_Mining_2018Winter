def predict(self, X):
        R, C = X.shape
        yhat = np.zeros(R)
        for m in range(R):
            if X[m, 0] > -80.305106:
                yhat[m] = 0
            else:
                if X[m, 1] > 37.669007:
                    yhat[m] = 0
                else:
                    yhat[m] = 1
        return yhat