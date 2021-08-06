import matplotlib.pyplot as plt
import copy
import numpy as np


class Stroke:
    def __print_points(self):
        for p in self.points_lst:
            print(p.x, p.y, p.t)
        print("")

    def __init__(self, lst):
        # reorder the list of points according to time
        self.points_lst = sorted(lst, key=lambda p: p.t)
        self.init_lst = copy.deepcopy(self.points_lst)
        ind = np.argmin(self.get_x())
        self.origin_x, self.origin_y = self.get_x()[ind], self.get_y()[ind]
        self.step_vector = []
        
    def __len__(self):
        return len(self.points_lst)

    def __eq__(self, other):
        if isinstance(other, Stroke):
            return self.len() == len(other) and all([x == y for x, y in zip(self.get_points(), other.get_points())])
        else:
            return False
    
    def __str__(self):
        return '[' + ', '.join([str(pt) for pt in self.get_points()]) + ']'

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

    def get_copy(self):
        pt_lst = []
        for p in self.get_points():
            pt_lst.append(p.get_copy())
        
        return Stroke(pt_lst)

    # for a given transformation parameters, transform all the points
    def transform(self, t, xo, yo, upd_step=True, retain_origin=False):
        if upd_step:
            self.step_vector = []
        for p in self.points_lst:
            x = (p.x - xo) * t[0] + (p.y - yo) * t[1] + t[2] 
            y = (p.x - xo) * t[3] + (p.y - yo) * t[4] + t[5]
            
            if retain_origin:
                x += xo
                y += yo

            if upd_step:
                self.step_vector.append((x - p.x, y - p.y))
            p.x, p.y = x, y

    # update step vector to prepare for the SketchAnimationing
    def upd_step_vector(self, t, xo, yo, retain_origin=False):
        self.step_vector = []
        for p in self.points_lst:
            x = (p.x - xo) * t[0] + (p.y - yo) * t[1] + t[2]
            y = (p.x - xo) * t[3] + (p.y - yo) * t[4] + t[5] 

            if retain_origin:
                x += xo
                y += yo
                
            self.step_vector.append((x - p.x, y - p.y))

