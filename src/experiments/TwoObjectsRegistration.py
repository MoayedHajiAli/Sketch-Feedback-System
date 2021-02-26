import sys
sys.path.insert(0, '../')

from register.Registration import RegisterTwoObjects, Registration
from animator.SketchAnimation import SketchAnimation
from utils.RegistrationUtils import RegistrationUtils
import matplotlib.pyplot as plt
import time
import numpy as np
import os.path as path

# enter input sample id
s = 9
# s = int(input())
ORG_FILE = 'input_directory/samples/test_samples/a' + str(s) + '.xml'
TAR_FILE = 'input_directory/samples/test_samples/b' + str(s) + '.xml'
ORG_FILE = path.join(path.abspath(path.join(__file__, '../../')), ORG_FILE)
TAR_FILE = path.join(path.abspath(path.join(__file__, '../../')), TAR_FILE)
reg = Registration(ORG_FILE, TAR_FILE, mn_stroke_len=3, re_sampling=0.5, flip=False, shift_target_y = 0)

# enter the indices of the original and target object
# a, b = map(int, input().split()) 
a, b = 2, 3

st = time.time()
obj1, obj2 = reg.original_obj[a], reg.target_obj[b]
print(obj1)
print("Orignal object length:", len(obj1))
print("Target object length:", len(obj2)) 
d, p = RegisterTwoObjects(obj1, obj2, reg.total_cost).optimize()
print("optimal cost:", d)
print("optimal transforamtion params:", [np.array(p)])
print("Running time: ", time.time()-st)

animation = SketchAnimation([obj1], [obj2])
# obj1.transform(RegistrationUtils.obtain_transformation_matrix(p))
animation.seq_animate_all([p], save=False, file="../test_videos/example7-obj3-4.mp4")
# plt.show()