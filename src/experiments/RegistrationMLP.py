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
import pathlib
import os.path as path
import matplotlib.pyplot as plt


# train a MLP for registration
dir = 'ASIST_Dataset/Data/Data_B/Circle'
dir = path.join(path.abspath(path.join(__file__ ,"../../..")), dir)
objs, labels = [], []
n, N = 0, 2500
for path in pathlib.Path(dir).iterdir():
    if n >= N:
        break
    a, b = ObjectUtil.xml_to_UnlabeledObjects(str(path))
    if n > 0:
        objs.extend(a)
        labels.extend(b)
    n += len(a)
print("Total size: {0}".format(n))
# print(len(objs))
# fig, axs = plt.subplots(4, 4)
# for i, obj in enumerate(objs):
#     obj.visualize(ax=axs[int(i/4), int(i%4)], show=False)

# plt.show()
registration_model(objs)
