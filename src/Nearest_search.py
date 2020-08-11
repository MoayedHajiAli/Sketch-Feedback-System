from scipy.spatial import cKDTree
import numpy as np
import warnings


class Nearest_search():

    def __init__(self, x, y):
        self.x_dia = max(x) - min(x)
        self.y_dia = max(y) - min(y)
        x = np.reshape(x, (len(x), 1))
        y = np.reshape(y, (len(y), 1))
        P = np.concatenate((x, y), axis=1)
        self.tree = cKDTree(P, leafsize=2)

    def _func(self, n, a, b):
        # 0.02 , 0.98 is the criteria where the y become meaningful in the function 2e^x / (1 + e^x) - 1
        c = np.log((1 + 0.02) / (1 - 0.02)) ** (1 / n)
        d = np.log((1 + 0.98) / (1 - 0.98)) ** (1 / n)
        def f(x):
            try:
                x2 = (x - a) * (d - c) / (b - a) + c
                if x2 >= 1:
                    return 1.0
                return (2 * (np.e ** (x2 ** n)) / (1 + np.e ** (x2 ** n))) - 1
            except RuntimeWarning:
                print(x, x2)
        return f

    def query(self, X, Y, step=10, mn_dis=2, mx_dis=100, fac=1000, dynamic=True, ration_mn=0.05, ration_mx=0.10):
        if dynamic:
            mn_dis = max(self.x_dia, self.y_dia) * ration_mn
            mx_dis = max(self.x_dia, self.y_dia) * ration_mx

        f = self._func(step, mn_dis, mx_dis)
        tot = 0.0
        for x, y in zip(X, Y):
            dd, ind = self.tree.query([[x, y]], k=1)
            res = f(dd[0])
            tot += fac * res
        return tot
