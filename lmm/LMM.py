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

    def nll(self):
        ym = self.y.T - self.X * self.beta
        V = self._v()
        [sign, ld] = np.linalg.slogdet(V)
        return 0.5 * ym.T * i(V) * ym + 0.5 * ld  # missing constant term

    def nll_d_sigma(self):
        V = self._v()
        inV = i(V)
        ym = self.y.T - self.X*self.beta

        # derivatives
        d_V_D = self.ZZt
        d_b_V = self._nll_b_d_V(inV)
        d_nll_b = -self.X.T
        d_nll_d = d_nll_b*d_b_V*d_V_D

        tmp1 = d_nll_d*V*ym + ym.T*d_V_D*ym + ym.T*V*d_nll_d
        tmp2 = i(V.T)*d_V_D

        return tmp1 + tmp2

    def _nll_b_d_V(self, inV):
        xvx = self.X.T*inV*self.X
        ixvx = i(xvx)
        xx = self.X.T*self.X
        tmp1 = np.trace((ixvx*xx*ixvx).T*inV*inV)*self.X.T*inV*self.y
        tmp2 = ixvx*inV.T*self.X.T*self.y*inV.T
        return tmp1 - tmp2

    def sgd(self):
        D = self._d()
        for epoch in range(self.epochs):
            grad = self.nll_d_sigma()
            D = D - self.step_size*grad


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
                self.sgd()
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
    lmm.train(X, Z, y, method='SGD')
    y_pred = lmm.predict(X, Z)
    shp = y.shape
    print len(np.where(y_pred.reshape(shp) == y)[0])/float(shp[0])