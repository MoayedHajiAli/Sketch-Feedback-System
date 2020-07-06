class Point():
    x = y = t = 0

    def __init__(self, x, y, t = -1.0):
        self.x = int(x)
        self.y = int(y)
        self.t = float(t)

    def __mul__(self, o):
        if isinstance(o, int) or isinstance(o, float):
            return Point(self.x * o, self.y * o, self.t)
        else:
            return Point(self.x * o.x, self.y * o.y, self.t)

