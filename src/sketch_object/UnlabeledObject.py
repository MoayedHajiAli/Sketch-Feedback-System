# hold a collection of strokes
import numpy as np
import matplotlib.pyplot as plt
import copy
from sketch_object.Stroke import Stroke


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

    def __eq__(self, other):
        if isinstance(other, UnlabeledObject):
            return len(self) == len(other) and all([x == y for x, y in zip(self.get_strokes(), other.get_strokes())])
        else:
            return False  

    def move_step(self, steps):
        for stroke in self.strokes_lst:
            stroke.move_step(steps)
  
    def visualize(self, show=True, axis=[], ax=plt.gca()):
        if len(axis) == 4:
            ax.axis = axis
        for stroke in self.strokes_lst:
            stroke.visualize(show=False, ax=ax)
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
    
    def get_t(self):
        tmp = []
        for stroke in self.strokes_lst:
            tmp.extend(stroke.get_t())
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
    def transform(self, t, upd_step=False, restore_origin=False):
        xo = yo = 0
        if restore_origin:
            xo, yo = self.origin_x, self.origin_y
        for stroke in self.strokes_lst:
            stroke.transform(t, xo, yo, upd_step=upd_step)

    # update step vector to prepare for the SketchAnimationing
    def upd_step_vector(self, t, restore_origin=False):
        xo = yo = 0
        if restore_origin:
            xo, yo = self.origin_x, self.origin_y
        for stroke in self.strokes_lst:
            stroke.upd_step_vector(t, xo, yo)
    
    def corresponding_stroke(self, ind):
        """for a given index, return the index of the stroke that contains this point

        Args:
            ind (int): the index of the point
        """
        for i in len(self.strokes_lst):
            if ind < len(self.strokes_lst[i]):
                return i
            ind -= len(self.strokes_lst[i])
        
        return -1
    
    def get_copy(self): 
        """get copy of the current object.
        """
        st_lst = []
        for st in self.strokes_lst:
            st_lst.append(st.get_copy())
        
        return UnlabeledObject(st_lst)
