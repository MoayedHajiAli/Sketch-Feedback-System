from Point import Point
import math

class Vector():
    p1 = p2 = -1

    def __init__(self, p1, p2):
        if isinstance(p1, float) or isinstance(p1, int):
            self.p1 = p1
            self.p2 = p2
        else:
            self.p1 = Point(0, 0)
            x = p2.x - p1.x
            y = p2.y - p1.y
            self.p2 = Point(x, y)

    def __mul__(self, o):
        if isinstance(o, int) or isinstance(o, float):
            return Vector(self.p1, Point(self.p2.x * o, self.p2.y * o))
        else:
            return Vector(self.p1.x * o.x, self.p2.y * o.y)

    def __len__(self):
        return math.sqrt((self.p2.x - self.p1.x)**2 + (self.p2.y - self.p1.y)**2)
