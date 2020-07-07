from Morph import Morph
from Registration import Registration
from ObjectUtil import ObjectUtil
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np





def main():
    morph = Morph('./test_samples/a.xml', './test_samples/b.xml')
    plt.close()
    #for obj in morph.original:
    #    obj.visualize()
    #morph.start()
    ObjectUtil.match_objects_size(morph.original[0], morph.target[0])
    print(len(morph.original[0]))
    print(len(morph.target[0]))
    reg = Registration(morph.target[0], morph.original[0])
    t = reg.optimize()

    fig = plt.figure()
    ax = fig.add_subplot()
    #ax.set_xlim([-1, 10])
    #ax.set_ylim([-10, 10])

    print("Original x before trans", morph.original[0].get_x())
    print("Original y before trans", morph.original[0].get_y())
    print("Final dissimilarity", reg.calc_dissimilarity(t, debug=False))

    morph.original[0].transform(t)
    morph.target[0].visualize(ax, show = False )
    morph.original[0].visualize(ax, show = False)

    print("Transformation matrix", t)
    print("Target x", morph.target[0].get_x())
    print("Target y", morph.target[0].get_y())
    print("Original x", morph.original[0].get_x())
    print("Original y", morph.original[0].get_y())
    plt.show()

if __name__ == '__main__':
    main()