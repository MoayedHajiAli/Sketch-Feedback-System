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

    # X, Y traverse for original objects
    x_traverse, y_traverse = [], []

    #figure
    fig = plt.figure()
    ax = plt.axes(xlim=(-100, 2000), ylim=(-100, 2000))

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

    # for a given two objects, calculate the x, y traverse for every step
    def calculate_path(self, obj1, obj2, steps=1000):
        # calculate start and end position
        mn_RMS, stx, sty, enx, eny = ObjectUtil.match_degree(obj1, obj2)
        print("Object", stx, sty, enx, eny)
        self.original_circles.append(plt.plot([stx], [sty], 'bo', ms=6)[0])
        self.target_circles.append(plt.plot([enx], [eny], 'bo', ms=6)[0])
        #self.ax.add_artist(self.original_circles[-1])
        #self.ax.add_artist(self.target_circles[-1])
        return (enx - stx) / steps, (eny - sty) / steps

    # for a given two set of objects, calculate the x, y traverse for every step
    # returns a list of X, Y
    def match(self, org_objs, tar_objs, steps = 1000):
        X, Y = [], []
        for obj1, obj2 in zip(org_objs, tar_objs):
            x, y = self.calculate_path(obj1, obj2, steps = steps)
            X.append(x)
            Y.append(y)
        return X, Y

    lines = []
    def init_animation(self):

        for pt, obj in zip(self.original_patches, self.original):
            obj.reset(), pt.set_data(obj.get_x(), obj.get_y())
        for pt, obj in zip(self.target_patches, self.target):
            obj.reset(), pt.set_data(obj.get_x(), obj.get_y())
        return self.original_patches + list(self.target_patches) + list(self.original_circles) + list(self.target_circles)

    def animate(self, i):
        if i == 19:
            self.init_animation()
        self.move_objects(self.original, self.x_traverse, self.y_traverse)
        for pt, obj, cir in zip(self.target_patches, self.target, self.original_circles):
            pt.set_data(obj.get_x(), obj.get_y())
        return self.original_patches + list(self.target_patches) + list(self.original_circles)

    def start(self):
        tem_org, tem_tar = [], []
        leb_org, leb_tar = [], []
        n = max(self.original_labels)
        for i in range(n+1):
            tmp = self.original_labels == i
            tem_org.extend(self.original[tmp])
            tem_tar.extend(self.target[self.target_labels == i])
            tem_x, tem_y = self.match(self.original[self.original_labels == i], self.target[self.target_labels == i], steps = 1000)
            self.x_traverse.extend(tem_x)
            self.y_traverse.extend(tem_y)

        # update order of objects of the orignal and target
        self.original, self.target = tem_org, tem_tar

        # update labels
        self.original_labels, self.target_labels = leb_org, leb_tar


        # for obj in self.original:
        #     obj.visualize()

        self.lines = [plt.plot([], [])[0] for _ in range(4)]
        print(self.x_traverse)
        # animate
        anim = animation.FuncAnimation(self.fig, func =self.animate,
                                       init_func = self.init_animation, frames = 10, interval = 20, blit = True)

        plt.show()