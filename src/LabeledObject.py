import matplotlib.pyplot as plt
import copy

class LabeledObject():

    # list of points
    lst = []
    init_lst = []

    def __print_points(self):
        for p in self.lst:
            print(p.x, p.y, p.t)
        print("")

    def __init__(self, lst):
        self.lst = sorted(lst, key = lambda p : p.t)
        self.init_lst = copy.deepcopy(self.lst)

    def __len__(self):
        return len(self.lst)

    def move(self, x, y):
        for p in self.lst:
            p.x += x
            p.y += y

    def visualize(self, axis = []):
        x = [pt.x for pt in self.lst]
        y = [pt.y for pt in self.lst]
        if len(axis) != 4:
            axis = [min(x)-50, max(x)+50, min(y)-50, max(y)+50]
        plt.axis = axis
        plt.plot(x, y)
        plt.show()

    def len(self):
        return len(self.lst)

    #get X coordinates of the points
    def get_x(self):
        return [p.x for p in self.lst]

    # get Y coordinates of the points
    def get_y(self):
        return [p.y for p in self.lst]

    def reset(self):
        self.lst = copy.deepcopy(self.init_lst)