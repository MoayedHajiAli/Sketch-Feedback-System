from scipy.spatial import cKDTree
from LabeledObject import LabeledObject
import numpy as np

class Nearest_search():

    def __init__(self, x, y):
        x = np.reshape(x, (len(x), 1))
        y = np.reshape(y, (len(y), 1))
        P = np.concatenate((x, y), axis=1)
        self.tree = cKDTree(P, leafsize=2)

    def query(self, X, Y):
        tot = 0.0
        for x, y in zip(X, Y):
            dd, ind = self.tree.query([[x, y]], k=1)
            tot += dd[0] ** 2
        return tot
