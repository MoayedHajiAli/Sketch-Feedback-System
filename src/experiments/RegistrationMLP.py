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
import time


# train a MLP for registration
FILE = '../input_directory/samples/test_samples/a3.xml'

objs, labels = ObjectUtil.xml_to_UnlabeledObjects(FILE, mn_len=5)
print(labels)
registration_model(objs)
