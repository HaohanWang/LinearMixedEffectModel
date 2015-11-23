__author__ = 'haohanwang'

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from scipy.special import expit


def i(X):
    return np.linalg.inv(X)


def p(X):
    return np.linalg.pinv(X)


# data = sm.datasets.get_rdataset('dietox', 'geepack').data
#
# print data.shape
#
# d = np.zeros((861, 7))
#
# i = 0
# for k in data:
#     d[:, i] = data[k]
#     i += 1
#
# print d
#
# md = smf.mixedlm('Weight ~ Feed', data=data, groups=data['Pig'])
# mdf = md.fit()
# print mdf.summary()

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

    def nll_d_beta(self):
        D = self._d()
        db = ((self.X * self.beta - self.y.T).T * self.Z * D * self.Z.T * self.X).T
        l1 = self.l1 * self.beta
        return db - l1

    def nll_d_mu(self):
        D = self._d()
        r = self.y.T - self.X * self.beta
        du = (r.T * self.ZZt * r + i(self.Z * D * self.Z.T).T) * self.mu.T
        l1 = self.l1 * self.mu
        return du - l1

    def sgd(self):
        nll_prev = self.neg_log_likelihood()
        for epoch in range(self.epochs):
            self.beta = self.beta + self.step_size * self.nll_d_beta()
            self.mu = self.mu + self.step_size * self.nll_d_mu()
            nll = self.neg_log_likelihood()
            if nll >= nll_prev:
                print 'Early Stop'
                break
            nll_prev = nll

    def _sigma(self):
        return (np.square(self.mu)) / self.q

    def m_step(self, inV):
        self.beta = p(self.X) * (self.X * self.beta + self.sig[0, 0] * self.X * i(self.X.T * self.X) * self.X.T * inV * (
        self.y.T - self.X * self.beta))

    def e_step(self, inV):
        r = self.y.T - self.X * self.beta
        self.sig = self.sig + np.square(self.sig) * (r.T * inV * self.ZZt * inV * r - np.trace(self.Z.T * inV * self.Z))
        # self.mu = np.sqrt(sig)

    def em(self):
        nll_prev = self.neg_log_likelihood()
        for epoch in range(self.epochs):
            self.sig = self._sigma()
            inV = i(self._v())
            self.m_step(inV)
            self.e_step(inV)
            nll = self.neg_log_likelihood()
            print nll
            if nll >= nll_prev:
                print 'Early Stop'
                break
            nll_prev = nll

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
                self.em()
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