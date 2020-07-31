# hold a collection of strokes
import numpy as np
import matplotlib.pyplot as plt
import copy


class UnlabeledObject:

    def __init__(self, strokes_lst):
        self.strokes_lst = strokes_lst
        # set a new origin
        ind = np.argmin(self.get_x())
        self.origin_x, self.origin_y = self.get_x()[ind], self.get_y()[ind]

    def print_strokes(self):
        for i, stroke in enumerate(self.strokes_lst):
            stroke.print_points()

    def __len__(self):
        return sum([len(stroke) for stroke in self.strokes_lst])

    def move_step(self, steps):
        for stroke in self.strokes_lst:
            stroke.move_step(steps)

    def visualize(self, show=True, axis=[], ax=plt.gca()):
        if len(axis) == 4:
            ax.axis = axis
        for stroke in self.strokes_lst:
            stroke.visualize(show=False, ax = plt.gca())
        if show:
            plt.show()

    def strokes_len(self):
        return len(self.strokes_lst)

    def get_strokes(self):
        return self.strokes_lst

    # get X coordinates of the points
    def get_x(self):
        tmp = []
        for stroke in self.strokes_lst:
            tmp.extend(stroke.get_x())
        return tmp

    # get x coordinates of point separated by strokes
    def get_strokes_x(self):
        tmp = []
        for stroke in self.strokes_lst:
            tmp.append(stroke.get_x())
        return tmp

    # get Y coordinates of the points
    def get_y(self):
        tmp = []
        for stroke in self.strokes_lst:
            tmp.extend(stroke.get_y())
        return tmp

    # get Y coordinates of the points separated by strokes
    def get_strokes_y(self):
        tmp = []
        for stroke in self.strokes_lst:
            tmp.append(stroke.get_y())
        return tmp

    def get_points(self):
        tmp = []
        for stroke in self.strokes_lst:
            tmp.extend(stroke.get_points())
        return tmp

    def reset(self):
        for stroke in self.strokes_lst:
            stroke.reset()

    # for a given transformation parameters, transform all the points
    def transform(self, t, upd_step=False, restore_origin=True):
        xo = yo = 0
        if restore_origin:
            xo, yo = self.origin_x, self.origin_y
        for stroke in self.strokes_lst:
            stroke.transform(t, xo, yo, upd_step=upd_step)

    # update step vector to prepare for the morphing
    def upd_step_vector(self, t, restore_origin=True):
        xo = yo = 0
        if restore_origin:
            xo, yo = self.origin_x, self.origin_y
        for stroke in self.strokes_lst:
            stroke.upd_step_vector(t, xo, yo)