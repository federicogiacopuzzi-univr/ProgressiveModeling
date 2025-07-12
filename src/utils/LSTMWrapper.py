from sklearn.base import BaseEstimator, RegressorMixin

def __init__(self, model, timesteps):
    self.model = model
    self.timesteps = timesteps

def fit(self, X, y):
    return self

def predict(self, X):
    X_seq = []
    for i in range(len(X) - self.timesteps):
        X_seq.append(X[i:i+self.timesteps])
    X_seq = np.array(X_seq)
    y_pred = self.model.predict(X_seq)
    return y_pred
