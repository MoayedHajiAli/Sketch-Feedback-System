import matplotlib.pyplot as plt
import copy


class Stroke:
    def __print_points(self):
        for p in self.points_lst:
            print(p.x, p.y, p.t)
        print("")

    def __init__(self, lst):
        self.points_lst = sorted(lst, key=lambda p: p.t)
        self.init_lst = copy.deepcopy(self.points_lst)
        self.step_vector = []

    def __len__(self):
        return len(self.points_lst)

    def move_step(self, steps):
        for p, v in zip(self.points_lst, self.step_vector):
            p.x += v[0]/steps
            p.y += v[1]/steps

    def visualize(self, show=True, axis=[], ax=plt.gca()):
        x = [pt.x for pt in self.points_lst]
        y = [pt.y for pt in self.points_lst]
        if len(axis) == 4:
            ax.axis = axis
        ax.plot(x, y)
        if show:
            plt.show()

    def len(self):
        return len(self.points_lst)

    # get X coordinates of the points
    def get_x(self):
        return [p.x for p in self.points_lst]

    # get Y coordinates of the points
    def get_y(self):
        return [p.y for p in self.points_lst]
    
    # get T of the points
    def get_t(self):
        return [p.t for p in self.points_lst]

    def get_points(self):
        return self.points_lst

    def reset(self):
        self.points_lst = copy.deepcopy(self.init_lst)

    # for a given transformation parameters, transform all the points
    def transform(self, t, xo, yo, upd_step=True):
        if upd_step:
            self.step_vector = []
        for p in self.points_lst:
            x = (p.x - xo) * t[0] + (p.y - yo) * t[1] + t[2] + xo
            y = (p.x - xo) * t[3] + (p.y - yo) * t[4] + t[5] + yo
            if upd_step:
                self.step_vector.append((x - p.x, y - p.y))
            p.x, p.y = x, y

    # update step vector to prepare for the morphing
    def upd_step_vector(self, t, xo, yo):
        self.step_vector = []
        for p in self.points_lst:
            x = (p.x - xo) * t[0] + (p.y - yo) * t[1] + t[2] + xo
            y = (p.x - xo) * t[3] + (p.y - yo) * t[4] + t[5] + yo
            self.step_vector.append((x - p.x, y - p.y))