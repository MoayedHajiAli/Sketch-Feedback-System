from scipy.spatial import cKDTree
import numpy as np
import warnings
from sketch_object.Point import Point

class NearestSearch():
    def __init__(self, x, y, step=10, mn_dis=10, mx_dis=1000, fac=1000, dynamic=True, ration_mn=0.0, ration_mx=0.6):
        x_dia = max(x) - min(x)
        y_dia = max(y) - min(y)
        x = np.reshape(x, (len(x), 1))
        y = np.reshape(y, (len(y), 1))
        self.P = np.concatenate((x, y), axis=1)
        self.tree = cKDTree(self.P, leafsize=2)

        if dynamic:
            mn_dis_x = x_dia * ration_mn
            mx_dis_x = x_dia * ration_mx
            mn_dis_y = y_dia * ration_mn
            mx_dis_y = y_dia * ration_mx
            
        self.f_x = self._func(step, mn_dis_x, mx_dis_x)
        self.f_y = self._func(step, mn_dis_y, mx_dis_y)

    def _func(self, n, a, b):
        # 0.02 , 0.98 is the criteria where the y become meaningful in the function 2e^x / (1 + e^x) - 1
        c = np.log((1 + 0.01) / (1 - 0.01)) ** (1 / n)
        d = np.log((1 + 0.99) / (1 - 0.99)) ** (1 / n)
        def f(x):
            try:
                x2 = (x - a) * (d - c) / (b - a) + c
                #TODO: why >= 40? where did 40 come from? 
                if (x2 ** n) >= 40:
                    return 1.0
                return (2 * (np.e ** (x2 ** n)) / (1 + np.e ** (x2 ** n))) - 1
            except RuntimeWarning:
                print(x, x2)
        return f

    def query(self, X, Y, fac = 1000):
        tot = 0.0
        for x, y in zip(X, Y):
            dd, ind = self.tree.query([[x, y]], k=1)
            x1, y1 = self.P[ind[0]][0], self.P[ind[0]][1]
            res = self.f_x(abs(x1 - x)) + self.f_y(abs(y1 - y))
            tot += fac * res
        return tot

    def query_ind(self, x, y):
        dd, ind = self.tree.query([[x, y]], k=1)
        return ind[0]

    def query_point(self, pPoint) -> Point:
        x, y = p.get_x(), p.get_y()
        dd, ind = self.tree.query([x, y], k = 1)
        return Point(self.P[ind[0]][0], self.P[ind[0]][1])