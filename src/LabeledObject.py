import matplotlib.pyplot as plt
import copy


class LabeledObject():

    def __print_points(self):
        for p in self.lst:
            print(p.x, p.y, p.t)
        print("")

    def __init__(self, lst):
        self.lst = sorted(lst, key=lambda p: p.t)
        self.init_lst = copy.deepcopy(self.lst)
        self.step_vector = []

    def __len__(self):
        return len(self.lst)

    def move_step(self, steps):
        #print(len(self.step_vector))
        for p, v in zip(self.lst, self.step_vector):
            p.x += v[0]/steps
            p.y += v[1]/steps

    def visualize(self, show=True, axis=[], ax=plt.gca()):
        x = [pt.x for pt in self.lst]
        y = [pt.y for pt in self.lst]
        if len(axis) == 4:
            ax.axis = axis
        ax.plot(x, y)
        if show:
            plt.show()

    def len(self):
        return len(self.lst)

    # get X coordinates of the points
    def get_x(self):
        return [p.x for p in self.lst]

    # get Y coordinates of the points
    def get_y(self):
        return [p.y for p in self.lst]

    def get_points(self):
        return self.lst

    def reset(self):
        self.lst = copy.deepcopy(self.init_lst)

    # for a given transformation parameters, transform all the points

    def transform(self, t):
        for p in self.lst:
            x = p.x * t[0] + p.y * t[1] + t[2]
            y = p.x * t[3] + p.y * t[4] + t[5]
            self.step_vector.append((x - p.x, y - p.y))
            p.x, p.y = x, y