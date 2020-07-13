import LabeledObject
from Point import Point
from Morph import Morph
from Registration import Registration
from ObjectUtil import ObjectUtil
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def main():
    morph = Morph('./test_samples/a.xml', './test_samples/b.xml')
    plt.close()
    # for obj in morph.original:
    #    obj.visualize()
    # morph.start()
    # test(morph.original[0])
    # return

    morph.original[1] = ObjectUtil.stroke_restructure(morph.original[1], 200)
    morph.target[1] = ObjectUtil.stroke_restructure(morph.target[1], 200)
    ObjectUtil.match_objects_size(morph.original[1], morph.target[1])
    reg = Registration(morph.target[1], morph.original[1])
    t = reg.optimize()
    fig = plt.figure()
    ax = fig.add_subplot()

    print("Original x before trans", morph.original[1].get_x())
    print("Original y before trans", morph.original[1].get_y())
    print(t)
    print("Final dissimilarity", reg.calc_dissimilarity(t))

    morph.original[1].visualize(ax=ax, show=False)
    morph.original[1].transform(t)
    morph.target[1].visualize(ax=ax, show=False)
    morph.original[1].visualize(ax=ax, show=False)

    print("Transformation matrix", t)
    print("Target x", morph.target[1].get_x())
    print("Target y", morph.target[1].get_y())
    print("Original x", morph.original[1].get_x())
    print("Original y", morph.original[1].get_y())
    plt.show()


def test(obj:LabeledObject):
    fig = plt.figure()
    ax = fig.add_subplot()
    obj.visualize(ax=ax, show = False)
    obj = ObjectUtil.stroke_restructure(obj, 400)
    print(obj.get_x())
    print(obj.get_y())
    obj.visualize(ax=ax, show = False)
    plt.show()

    print(len(obj))
    for i in range(len(obj)-1):
        print(Point.euclidean_distance(obj.get_points()[i], obj.get_points()[i + 1]))


if __name__ == '__main__':
    main()
