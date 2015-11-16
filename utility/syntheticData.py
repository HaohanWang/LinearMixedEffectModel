__author__ = 'haohanwang'

import numpy as np
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


n = 1000
d = 100
e = 50

beta = np.random.random(d)
beta[np.where(beta<=0.5)] = 0
beta = np.matrix(beta)
mu = np.matrix(np.random.random(50))
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

save_object(data, '../data/synthetic.pkl', )