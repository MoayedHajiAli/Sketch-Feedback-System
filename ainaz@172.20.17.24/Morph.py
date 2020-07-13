from Point import Point
from LabeledObject import LabeledObject
from ObjectUtil import ObjectUtil
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


class Morph():
    original, target = [], []
    original_labels = np.array([0, 0, 0, 1, 2, 2])
    target_labels = np.array([0, 0, 0, 1, 2, 2])

    # plots for animating the original objects
    original_patches, target_patches = np.array([]), np.array([])
    original_circles, target_circles = [], []
    original_circles_coords = []

    # X, Y traverse for original objects
    x_traverse, y_traverse = [], []

    # figure
    fig = plt.figure()
    ax = plt.axes(xlim=(-100, 2500), ylim=(-100, 1500))

    def __init__(self, org_file, tar_file):
        self.original = ObjectUtil.xml_to_LabledObjects(org_file)
        self.target = ObjectUtil.xml_to_LabledObjects(tar_file)
        self.original_patches = [plt.plot([], [])[0] for _ in range(len(self.original))]
        self.target_patches = [plt.plot([], [])[0] for _ in range(len(self.target))]

    # update the coordinates of the all the objects
    # lst: list of objects, x, y: list of x, y traverse for objects
    def move_objects(self, lst, x, y):
        for obj, x, y in zip(lst, x, y):
            obj.move(x, y)

    # for a given two objects, match the object by trying all pair of points
    # and calculate the x, y traverse
    def find_single_translation_path(self, obj1, obj2, steps=1000):
        # calculate start and end position
        mn_RMS, stx, sty, enx, eny = ObjectUtil.find_optimal_translation(obj1, obj2)
        self.original_circles.append(plt.plot([], [], 'bo', ms=6)[0])
        self.original_circles_coords.append([stx, sty])
        self.target_circles.append(plt.plot([enx], [eny], 'bo', ms=6)[0])
        return (enx - stx) / steps, (eny - sty) / steps

    # for a given two set of objects, calculate the x, y traverse for every step
    # returns a list of X, Y
    def find_translation_paths(self, org_objs, tar_objs, steps=1000):
        X, Y = [], []
        for obj1, obj2 in zip(org_objs, tar_objs):
            x, y = self.find_single_translation_path(obj1, obj2, steps=steps)
            X.append(x)
            Y.append(y)
        return X, Y

    # initial function for animation
    def init_animation(self):
        for pt, obj in zip(self.original_patches, self.original):
            obj.reset(), pt.set_data(obj.get_x(), obj.get_y())
        for pt, obj in zip(self.target_patches, self.target):
            obj.reset(), pt.set_data(obj.get_x(), obj.get_y())
        return self.original_patches + list(self.target_patches) + list(self.original_circles) + list(
            self.target_circles)

    # animate function for matplotlib animation. It is being called in every step
    def animate(self, i):
        self.move_objects(self.original, self.x_traverse, self.y_traverse)

        j = 0
        for pt, obj, cir in zip(self.original_patches, self.original, self.original_circles):
            pt.set_data(obj.get_x(), obj.get_y())
            self.original_circles_coords[j][0] += self.x_traverse[j]
            self.original_circles_coords[j][1] += self.y_traverse[j]
            cir.set_xdata([self.original_circles_coords[j][0]])
            cir.set_ydata([self.original_circles_coords[j][1]])
            j += 1

        return self.original_patches + list(self.target_patches) + list(self.original_circles)

    def start_translation_morphing(self):
        tem_org, tem_tar = [], []
        leb_org, leb_tar = [], []
        n = max(self.original_labels)
        for i in range(n + 1):
            tmp = self.original_labels == i
            tem_org.extend(self.original[tmp])
            tem_tar.extend(self.target[self.target_labels == i])
            tem_x, tem_y = self.find_translation_paths(self.original[self.original_labels == i],
                                                       self.target[self.target_labels == i], steps=200)
            self.x_traverse.extend(tem_x)
            self.y_traverse.extend(tem_y)

        # update order of objects of the original and target
        self.original, self.target = tem_org, tem_tar

        # update labels
        self.original_labels, self.target_labels = leb_org, leb_tar

        # animate
        anim = animation.FuncAnimation(self.fig, func=self.animate,
                                       init_func=self.init_animation, frames=200, interval=20, blit=True,
                                       repeat_delay=2000, repeat=False)

        plt.show()
