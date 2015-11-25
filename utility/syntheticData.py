__author__ = 'haohanwang'

import numpy as np
import pickle
from matplotlib import pyplot as plt

from scipy.special import expit
from scipy.stats import logistic

def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return np.array(s)


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def original():
    n = 1000
    d = 100
    e = 50

    beta = np.random.random(d)
    beta[np.where(beta<=0.5)] = 0
    beta = np.matrix(beta)
    mu = np.matrix(np.random.random(e))
    D = np.std(mu)

    X = np.matrix(np.random.random((n, d)))
    Z = np.matrix(np.random.random((n, e)))
    y = np.zeros(n)
    for i in range(n):
        y[i] = np.random.normal(X[i,:]*beta.T, Z[i,:]*D*Z[i,:].T)

    m = np.mean(y)
    y[np.where(y<=m)] = 0
    y[np.where(y>m)] = 1

    data = {}
    data['X'] = X
    data['Z'] = Z
    data['y'] = y

    y_pred = sigmoid(X*beta.T)
    ind = np.where(y_pred<=0.5)
    y_pred[ind] = 0
    ind = np.where(y_pred>0.5)
    y_pred[ind] = 1
    shp = y.shape
    print len(np.where(y_pred.reshape(shp) == y)[0])/float(shp[0])

    # save_object(data, '../data/synthetic.pkl', )

def new_smaller():
    n = 100
    d = 10
    e = 2

    beta = np.random.random(d)
    beta[np.where(beta<=0.5)] = 0
    beta = np.matrix(beta)
    mu = np.matrix(np.random.random(e))
    D = np.std(mu)

    X = np.matrix(np.random.random((n, d)))
    Z = np.matrix(np.random.random((n, e)))

    y = X*beta.T

    m = np.mean(y)
    y[np.where(y<=m)] = 0
    y[np.where(y>m)] = 1


    y_pred = sigmoid(X*beta.T)
    # print y_pred
    m = np.mean(y_pred)
    ind = np.where(y_pred<=m)
    y_pred[ind] = 0
    ind = np.where(y_pred>m)
    y_pred[ind] = 1
    shp = y.shape
    print len(np.where(y_pred.reshape(shp) == y)[0])/float(shp[0])

def existing():
    import statsmodels.api as sm
    data = sm.datasets.get_rdataset('dietox', 'geepack').data
    weights = data['Weight']
    w = []
    t = []
    p = []
    for i in range(len(weights)):
        w.append(weights[i])
        t.append(data['Time'][i])
        p.append(data['Pig'][i])

    # y = np.array(w)
    # X = np.matrix(np.array(t))
    # Z = np.matrix(np.array(p))
    # data = {}
    # data['X'] = X.T
    # data['Z'] = Z.T
    # data['y'] = y.T
    # save_object(data, '../data/pig.pkl', )
    plt.scatter(t, w)
    plt.show()


if __name__ == '__main__':
    existing()