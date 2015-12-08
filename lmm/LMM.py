__author__ = 'haohanwang'

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from scipy.special import expit
from utility.dataManager import PLinkFormatReader as PLFR


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
        self.D = None

    def _d(self):
        return np.diag(np.square(self.mu.T.tolist()[0])) # self.mu.T*self.mu

    def _v(self):
        return self.Z * self.D * self.Z.T

    def log_likelihood(self):
        ym = self.y.T - self.X * self.beta
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
        ym = self.y.T - self.X * self.beta

        # derivatives
        d_V_D = self.ZZt
        d_b_V = self._nll_b_d_V(inV)
        d_nll_b = -self.X.T
        d_nll_d = d_nll_b * d_b_V * d_V_D

        tmp1 = d_nll_d * V * ym + ym.T * d_V_D * ym + ym.T * V * d_nll_d
        tmp2 = i(V.T) * d_V_D

        return tmp1 + tmp2

    def _nll_b_d_V(self, inV):
        xvx = self.X.T * inV * self.X
        ixvx = i(xvx)
        tmp1 = ixvx * self.X.T * inV * inV * self.X * ixvx * self.X.T * inV * self.y.T
        tmp2 = ixvx * self.X.T * inV.T * inV.T * self.y.T
        return tmp1 - tmp2

    def sgd(self):
        D = self._d()
        for epoch in range(self.epochs):
            grad = self.nll_d_sigma()
            D = D - self.step_size * grad

    # def em(self):
    #     V = self._v()
    #     prev_ll = self.log_likelihood()
    #     print prev_ll
    #     for epoch in range(self.epochs):
    #         # ym = self.y.T - self.X*self.beta
    #         #grad = self.Z.T*ym*ym.T*self.Z + self.Z.T*(i(V.T))*self.Z
    #         self.D = self._d()
    #         V = self._v()
    #         inV = i(V)
    #         self.beta = i(self.X.T*inV*self.X)*self.X.T*inV*self.y.T
    #         self.mu = self.D*self.Z.T*inV*(self.y.T - self.X*self.beta)
    #         # print self.beta.mean()
    #         ll = self.log_likelihood()[0,0]
    #         print ll


    def em(self):
        self.D = (self.mu.T*self.mu/self.q)[0,0]
        prev_ll = self.log_likelihood()
        prev_ll = -float('inf')
        print prev_ll
        xb = self.X*self.beta
        ixx = i(self.X.T*self.X)
        V = self.ZZt
        inV = i(V)
        ym = self.y.T - self.X*self.beta
        for epoch in range(self.epochs):
            xb += self.D*self.X*ixx*self.X.T*inV*ym
            ym = self.y.T - xb
            self.D += self.D**2/self.q * ((ym.T*inV*self.ZZt*inV*ym) - np.trace(self.Z.T*inV*self.Z))
            self.D = self.D[0,0]
            print self.D
            curr_ll = self.log_likelihood()
            print curr_ll
            if curr_ll < prev_ll:
                break
            prev_ll = curr_ll


    def em_multi_variance(self):
        prev_ll = self.log_likelihood()
        print prev_ll
        sigma = np.square(self.mu)
        sig = sigma.T.tolist()[0]
        zshp = self.Z.shape

        V = self._v()
        inV = i(V)
        Py = None
        for epoch in range(self.epochs):
            P = inV - inV * self.X * i(self.X.T * inV * self.X) * self.X.T * inV
            yP = self.y * P
            Py = P * self.y.T

            for j in range(zshp[1]):
                sig[j] += (sig[j] ** 2) * \
                          (yP * self.Z[:, j] * self.Z[:, j].T * Py - np.trace(self.Z[:, j].T * inV * self.Z[:, j]))[
                              0, 0]
            self.D = np.diag(sig)
            V = self._v()
            inV = i(V)
            ll = self.log_likelihood()
            print ll
            # print sig
        self.beta = i(self.X.T * inV * self.X) * self.X.T * inV * self.y.T
        self.mu = self.D * self.Z.T * Py

    def classify(self, X, Z):
        y_pred = expit(X * self.beta)
        m = np.mean(y_pred)
        ind = np.where(y_pred <= m)
        y_pred[ind] = 0
        ind = np.where(y_pred > m)
        y_pred[ind] = 1
        return y_pred

    def regress(self, X, Z):
        return X*self.beta

    def train(self, X, Z, y, method='EM', cost='ML', epochs=50, step_size=1):
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
        self.mu = np.matrix(np.random.random((shp2[1], 1))) - 0.5
        self.q = shp2[1]
        self.epochs = epochs
        self.step_size = step_size
        self.D = self._d()
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
    plr = PLFR('../data/sampleData/')
    X, Z, y = plr.readFile('geno_test.tped', 'geno_cov.tped', 'pheno.txt', phenoCol=2)
    print X.shape
    print Z.shape
    print y.shape

    lmm = LMM()
    lmm.train(X, Z, y, method='EM')
    y_pred = lmm.classify(X, Z)
    # print y_pred
    shp = y.shape
    print len(np.where(y_pred.reshape(shp) == y)[0]) / float(shp[0])
    print y_pred
    print y
    # print np.square(y-y_pred).mean()