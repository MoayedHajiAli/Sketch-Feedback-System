from scipy.spatial import cKDTree
import numpy as np

class Nearest_search():

    def __init__(self, x, y):
        x = np.reshape(x, (len(x), 1))
        y = np.reshape(y, (len(y), 1))
        P = np.concatenate((x, y), axis=1)
        self.tree = cKDTree(P, leafsize=2)

    def _func(self, n, a, b):
        # 0.02 , 0.98 is the criteria where the y become meaningful in the function 2e^x / (1 + e^x) - 1
        c = np.log((1 + 0.02) / (1 - 0.02)) ** (1 / n)
        d = np.log((1 + 0.98) / (1 - 0.98)) ** (1 / n)
        def f(x):
            x2 = (x - a) * (d - c) / (b - a) + c
            return (2 * (np.e ** x2) / (1 + np.e ** x2)) - 1
        return f

    def query(self, X, Y, step=4, mn_dis=4, mx_dis=50, fac=1000):
        f = self._func(step, mn_dis, mx_dis)
        tot = 0.0
        for x, y in zip(X, Y):
            dd, ind = self.tree.query([[x, y]], k=1)
            tot += fac * f(dd[0])
        return tot
