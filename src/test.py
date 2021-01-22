
# import numpy as np
# from matplotlib import pyplot as plt
# from utils.ObjectUtil import ObjectUtil
from sketch_object.UnlabeledObject import UnlabeledObject
from sketch_object.Point import Point
from sketch_object.Stroke import Stroke


def test_eq():
        obj1 = UnlabeledObject([Stroke([Point(1, 2, 3), Point(0, 0, 1)])])
        obj2 = UnlabeledObject([Stroke([Point(1, 2, 0), Point(1, 0, 1)])])
        if(obj1 == obj2):
                print("equal")



