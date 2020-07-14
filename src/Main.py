from LabeledObject import LabeledObject
from Point import Point
from Morph import Morph
from Registration import Registration
from ObjectUtil import ObjectUtil
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def main():
    #morph = Morph('./test_samples/a1.xml', './test_samples/b1.xml')
    # for obj in morph.original:
    #    obj.visualize()
    #morph.animate_translation_all(1000)
    # test(morph.original[0])
    # return

    reg = Registration('./test_samples/a1.xml', './test_samples/b1.xml')
    #print(len(reg.original), len(reg.target))
    #t = reg.register()
    t = [ [ 1.15486147e+00 , 1.30689773e-01 ,-9.01251974e+02 ,-2.03297045e-01,
  5.45067943e-01, -1.77410387e+02],[-5.37882914e-01,  5.38213965e-01,  8.83387451e+02 ,-7.05033356e-01,
 -1.77113356e-01,  1.09829809e+03],[ 5.94574881e-01 , 5.66680239e-01 ,-9.31690734e+02 , 7.73525398e-01,
 -2.81408830e-01, -7.44952052e+02],[-8.59521477e-01,  3.09320939e-02,  1.76955465e+03 ,-2.07680950e-01,
 -8.24981297e-01 , 1.47310151e+03],[ 8.94498836e-01, -2.77762901e-01 ,-1.14543000e+03 , 1.28229238e-01,
  8.07712227e-01, -6.59892149e+02],[-9.12940478e-01,  9.54378559e-02 , 2.82242401e+03 , 1.41318641e-01,
 -7.95573252e-01,  7.98774326e+02] ]

    t2 = [[7.13886756e-01,-4.88776254e-01,6.69453413e+02,4.84104502e-01,3.93501924e-01,-2.41914235e+02],[7.31576612e-01,-2.24359612e-02,3.12301042e+02,1.40973402e-01,6.13162765e-01,-1.52914604e+02],[-5.37672730e-01,1.56887336e-01,2.35072254e+03,8.43622183e-01,-6.69032453e-01,-2.43154201e+02],[-1.04228371e+00,3.02001652e+00,7.64406041e+02,-6.98929400e-01,1.22309348e+00,3.96434017e+02],[8.87176278e-01,-2.04374310e-01,1.01597127e+02,-8.09974192e-02,-7.06976350e-01,1.06360104e+03],[5.07812064e-01,4.84988516e-01,2.78593778e+02,7.29621852e-01,-4.67259337e-01,-5.27188628e+02],[8.43031875e-01,-5.74740101e-01,4.15820106e+02,2.66369116e-01,2.72194165e-02,-1.44648304e+02],[8.93363298e-02,5.67784991e+00,-2.92845644e+03,-7.49535573e-03,-6.80986296e-01,9.53719689e+02],[3.72872770e-01,3.31824457e-01,7.45737594e+02,3.38756956e+00,3.08667179e+00,-8.59701405e+03],[1.82185708e+00,-2.34006048e+00,-8.42731218e+02,-3.41370770e-02,4.81108398e-02,4.68178208e+02]]

    morph = Morph('./test_samples/a1.xml', './test_samples/b1.xml')
    morph.animate_all(t)


def print_lst(lst):
    st = ','.join(map(str, lst))
    print('[', st, ']')

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
