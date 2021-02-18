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
from sklearn.model_selection import train_test_split



# # train a MLP for registration
# dir = 'ASIST_Dataset/Data/Data_B/Circle'
# dir = path.join(path.abspath(path.join(__file__ ,"../../..")), dir)
# objs, labels, val_objs, val_labels = [], [], [], []
# n, N, val_N = 0, 400, 0
# for file_path in pathlib.Path(dir).iterdir():
#     if n >= val_N:
#         break
#     a, b = ObjectUtil.xml_to_UnlabeledObjects(str(file_path))
#     if n > N:
#         val_objs.extend(a)
#         val_labels.extend(b)
#     elif n > 400:
#         objs.extend(a)
#         labels.extend(b)
#     n += len(a)

# # train a MLP for registration
# dir = 'ASIST_Dataset/Data/Data_B/Triangles'
# dir = path.join(path.abspath(path.join(__file__ ,"../../..")), dir)
# n, N, val_N = 0, 1000, 1050
# for file_path in pathlib.Path(dir).iterdir():
#     if n >= val_N:
#         break
#     a, b = ObjectUtil.xml_to_UnlabeledObjects(str(file_path))
#     if n > N:
#         val_objs.extend(a)
#         val_labels.extend(b)
#     elif n > 1000:
#         objs.extend(a)
#         labels.extend(b)
#     n += len(a)
# print("Total size: {0}".format(n))
# print(len(objs))
# fig, axs = plt.subplots(4, 4)
# for i, obj in enumerate(objs):
#     obj.visualize(ax=axs[int(i/4), int(i%4)], show=False)

dir = 'ASIST_Dataset/Data/Data_A/'
dir = path.join(path.abspath(path.join(__file__ ,"../../..")), dir)
K = 100
org_objs, tar_objs = [], []
objs, labels = ObjectUtil.extract_objects_from_directory(dir, n_files=5000, labels=['circle'])
labels = np.asarray(labels)
objs = np.asarray(objs)
for obj, lbl in zip(objs, labels):
    matched_objs = objs[labels == lbl]
    matched_objs = matched_objs[:min(K, len(matched_objs))]
    for obj2 in matched_objs:
        org_objs.append(obj)
        tar_objs.append(obj2)

# split train test
train_org_sketches, val_org_sketches, train_tar_sketches, val_tar_sketches = train_test_split(org_objs, tar_objs, test_size=0.2)
print(len(train_org_sketches), len(train_tar_sketches))
# plt.show()
registration_model(train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches)
