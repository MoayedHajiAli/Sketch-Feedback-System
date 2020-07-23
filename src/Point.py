import numpy as np


class Point():
    x = y = t = 0

    def __init__(self, x, y, t=-1.0):
        self.x = float(x)
        self.y = float(y)
        self.t = float(t)

    def __mul__(self, o):
        if isinstance(o, int) or isinstance(o, float):
            return Point(self.x * o, self.y * o, self.t)
        else:
            return Point(self.x * o.x, self.y * o.y, self.t)

    # for a given two points, find their euclidean distance
    @staticmethod
    def euclidean_distance(p1, p2):
        return np.sqrt((p2.y - p1.y) ** 2 + (p2.x - p1.x) ** 2)
