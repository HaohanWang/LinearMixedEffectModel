__author__ = 'haohanwang'

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from scipy.special import expit


def i(X):
    return np.linalg.inv(X)


def p(X):
    return np.linalg.pinv(X)

class LMM:
    def __init__(self, l1=0):
        self.l1 = l1
        self.beta = None
        self.mu = None
        self.X = None
        self.Z = None
        self.y = None
        self.ZZt = None  # Z * Z.T
        self.q = None  # dimension of mu
        self.sig = None

    def _d(self):
        return np.diag(np.square(self.mu.T.tolist()[0]))

    def _v(self):
        D = self._d()
        return self.Z * D * self.Z.T

    def log_likelihood(self):
        ym = self.y - self.y.T - self.X * self.beta
        V = self._v()
        [sign, ld] = np.linalg.slogdet(V)
        return - 0.5 * ym.T * i(V) * ym - 0.5 * ld  # missing constant term

    def neg_log_likelihood(self):
        ym = self.y.T - self.X * self.beta
        V = self._v()
        [sign, ld] = np.linalg.slogdet(V)
        return 0.5 * ym.T * i(V) * ym + 0.5 * ld + self.l1 * (
        np.abs(self.beta.sum()) + np.abs(self.mu.sum()))  # missing constant term





    def predict(self, X, Z):
        return expit(X * self.beta + Z * self.mu)

    def train(self, X, Z, y, method='EM', cost='ML', epochs=1000, step_size=1):
        '''
        :param X: fixed effect input
        :param Z: random effect input
        :param method: learning method, EM, SGD, Newton
        :param cost: ML, REML
        :return:
        '''
        shp1 = X.shape
        shp2 = Z.shape
        self.X = np.matrix(X)
        self.Z = np.matrix(Z)
        self.y = np.matrix(y)
        self.ZZt = Z * Z.T
        np.random.seed(0)
        self.beta = np.matrix(np.random.random((shp1[1], 1)))
        np.random.seed(0)
        self.mu = np.matrix(np.random.random((shp2[1], 1)))
        self.q = shp2[1]
        self.epochs = epochs
        self.step_size = step_size
        if cost == 'ML':
            if method == 'SGD':
                pass
            elif method == 'EM':
                pass
            elif method == 'Newton':
                pass
            else:
                print 'ERROR'
                return None
        elif cost == 'REML':
            pass
        else:
            print 'ERROR'
            return None


if __name__ == '__main__':
    import pickle

    data = pickle.load(open('../data/synthetic.pkl'))
    X = data['X']
    Z = data['Z']
    y = data['y']
    lmm = LMM()
    lmm.train(X, Z, y, method='EM')
    y_pred = lmm.predict(X, Z)
    shp = y.shape
    print len(np.where(y_pred.reshape(shp) == y)[0])/float(shp[0])