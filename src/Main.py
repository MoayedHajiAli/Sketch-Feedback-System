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
    ind1 = 2
    ind2 = 3
    morph.original[ind1] = ObjectUtil.stroke_restructure(morph.original[ind1], 200)
    morph.target[ind2] = ObjectUtil.stroke_restructure(morph.target[ind2], 200)
    ObjectUtil.match_objects_size(morph.original[ind1], morph.target[ind2])
    reg = Registration(morph.target[ind2], morph.original[ind1])
    # t = reg.optimize()
    t = np.array([-8.93154420e-01,  3.41597988e-01,  1.26656397e+03 , 4.09735746e-01,
  6.25855296e-01, -8.38676869e+02])
    print(t)
    fig = plt.figure()
    ax = fig.add_subplot()

    print("Original x before trans", morph.original[ind1].get_x())
    print("Original y before trans", morph.original[ind1].get_y())
    print(t)
    print("Final dissimilarity", reg.calc_dissimilarity(t))

    morph.original[ind1].visualize(ax=ax, show=False)
    morph.original[ind1].transform(t)
    morph.target[ind2].visualize(ax=ax, show=False)
    morph.original[ind1].visualize(ax=ax, show=False)

    print("Transformation matrix", t)
    print("Target x", morph.target[ind2].get_x())
    print("Target y", morph.target[ind2].get_y())
    print("Original x", morph.original[ind1].get_x())
    print("Original y", morph.original[ind1].get_y())
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
