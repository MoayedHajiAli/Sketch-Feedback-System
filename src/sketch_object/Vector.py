from sketch_object.Point import Point
import numpy as np


class Vector():
    x = y = -1

    def __init__(self, p1, p2):
        if isinstance(p1, float) or isinstance(p1, int):
            self.x = p1
            self.y = p2
        else:
            self.p1 = Point(0, 0)
            self.x = p2.x - p1.x
            self.y = p2.y - p1.y
            self.p2 = Point(self.x, self.y)

    def __mul__(self, o):
        if isinstance(o, int) or isinstance(o, float):
            return Vector(self.p1, Point(self.p2.x * o, self.p2.y * o))
        else:
            return self.p1.x * o.x + self.p2.y * o.y

    def __len__(self):
        return np.sqrt((self.p2.x - self.p1.x) ** 2 + (self.p2.y - self.p1.y) ** 2)

    @staticmethod
    def det(v1, v2):
        return v1.x * v2.y - v1.y * v2.x
