from Point import Point
from LabeledObject import LabeledObject
from ObjectUtil import ObjectUtil
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


class Morph():
    original_labels = np.array([0, 0, 0, 1, 2, 2])
    target_labels = np.array([0, 0, 0, 1, 2, 2])

    fig, ax = plt.gcf(), plt.gca()

    def __init__(self, org_file, tar_file):
        self.original = ObjectUtil.xml_to_LabledObjects(org_file)
        self.target = ObjectUtil.xml_to_LabledObjects(tar_file)
        self.original_patches = [plt.plot([], [])[0] for _ in range(len(self.original))]
        self.target_patches = [plt.plot([], [])[0] for _ in range(len(self.target))]

        # figure
        self.fig, self.ax = plt.gcf(), plt.gca()

    # update the coordinates of the all the objects
    # lst: list of objects, x, y: list of x, y traverse for objects
    def move_all(self, lst, steps):
        for obj in lst:
            obj.move_step(steps)

    # for a given two objects, match the object by trying all pair of points
    # and calculate the x, y traverse
    def find_single_translation_path(self, obj1, obj2):
        # calculate start and end position
        mn_RMS, stx, sty, enx, eny = ObjectUtil.find_optimal_translation(obj1, obj2)
        # self.original_circles.append(plt.plot([], [], 'bo', ms=6)[0])
        # self.original_circles_coords.append([stx, sty])
        # self.target_circles.append(plt.plot([enx], [eny], 'bo', ms=6)[0])
        return (enx - stx), (eny - sty)

    # for a given two set of objects, calculate the x, y traverse for every step
    # returns a list of X, Y
    def find_translation_paths(self, org_objs, tar_objs):
        X, Y = [], []
        for obj1, obj2 in zip(org_objs, tar_objs):
            x, y = self.find_single_translation_path(obj1, obj2)
            X.append(x)
            Y.append(y)
        return X, Y

    # initial function for animation
    def _init_animation(self):
        self.ax.set_xlim(0, 3000)
        self.ax.set_ylim(-100, 1200)
        for obj in self.original:
            obj.reset()
        for pt, obj in zip(self.original_patches, self.original):
            obj.reset, pt.set_data(obj.get_x(), obj.get_y())
        for pt, obj in zip(self.target_patches, self.target):
            obj.reset, pt.set_data(obj.get_x(), obj.get_y())
        return self.original_patches + list(self.target_patches)

    # animate function for matplotlib animation. It is being called in every step
    def _animate(self, i, steps):
        self.move_all(self.original, steps)
        for pt, obj in zip(self.original_patches, self.original):
            pt.set_data(obj.get_x(), obj.get_y())

        return self.original_patches + list(self.target_patches)

    def animate_translation_all(self, steps=1000):
        tem_org, tem_tar = [], []
        leb_org, leb_tar = [], []
        x_tra, y_tra = [], []
        n = max(self.original_labels)
        for i in range(n + 1):
            tem_org.extend(self.original[self.original_labels == i])
            tem_tar.extend(self.target[self.target_labels == i])
            tem_x, tem_y = self.find_translation_paths(self.original[self.original_labels == i],
                                                       self.target[self.target_labels == i])
            x_tra.extend(tem_x)
            y_tra.extend(tem_y)

        # update order of objects of the original and target
        self.original, self.target = tem_org, tem_tar

        # update labels
        self.original_labels, self.target_labels = leb_org, leb_tar

        # translate objects to obtain morphing step vectors then reset
        for obj, x, y, in zip(self.original, x_tra, y_tra):
            obj.transform([1, 0, x, 0, 1, y])
            obj.reset()

        self.animate(steps=steps)

    def animate(self, steps=1000):
        # animate
        anim = animation.FuncAnimation(self.fig, func=self._animate,
                                       init_func=self._init_animation, frames=steps, interval=1, blit=True,
                                       repeat_delay=2000, fargs=[steps])
        self._save_anim(anim, "example2.mp4")
        plt.show()
        return anim

    def animate_all(self, trans_matrix, steps=1000):
        # transform all objects to obtain morphing step vectors then reset them to their original coordinates
        for obj, t in zip(self.original, trans_matrix):
            obj.transform(t)
            obj.reset()
        self.animate()

    def _save_anim(self, anim, name):
        # Set up formatting for the movie files
        anim.save(name, writer='ffmpeg')