import sys
sys.path.insert(0, '../')

from animator.SketchAnimation import SketchAnimation
from register.Registration import Registration, RegisterTwoObjects
from matplotlib import pyplot as plt
import numpy as np
from utils.RegistrationUtils import RegistrationUtils
from utils.ObjectUtil import ObjectUtil
import copy
from sketch_object.UnlabeledObject import UnlabeledObject
from sketch_object.Stroke import Stroke
from tools.ClassEvaluation import ClassEvaluation
from tools.ObjectParsing import ObjectParsing
from tools.StrokeClustering import DensityClustering
from tools.ParsingEvaluation import ParsingEvaluation
from registrationNN.models import registration_model
import unittest


class ObjectUtilsTest(unittest.TestCase):
    FILE = '../input_directory/samples/test_samples/a3.xml'
    def setUp(self):
        # import original and target sketch as UnlabeledObjects
        self.objs, self.labels = ObjectUtil.xml_to_UnlabeledObjects(self.FILE, mn_len=5)

    def test_accumulative_to_poly(self):
        # convert to accumulative stroke3
        sketches = ObjectUtil.poly_to_accumulative_stroke3(self.objs)
        # convert back to poly
        sketches = ObjectUtil.accumalitive_stroke3_to_poly(sketches)
        
        # viualize first sketch from objects and transformed objects
        # please note that objects coordinates will be normalizes and thus will change. 
        # However, they should still represent the same object
        self.objs[0].visualize()
        sketches[0].visualize()

if __name__ == '__main__':
    unittest.main()