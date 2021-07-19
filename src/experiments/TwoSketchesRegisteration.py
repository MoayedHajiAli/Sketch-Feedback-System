import sys
sys.path.insert(0, '../')

from register.Registration import Registration
from animator.SketchAnimation import SketchAnimation
import matplotlib.pyplot as plt
import time
import copy
from sketch_object.UnlabeledObject import UnlabeledObject
from sketch_object.Stroke import Stroke
from utils.RegistrationUtils import RegistrationUtils
import numpy as np
import os.path as path

def add_objects(reg, lst):
    for ind in lst:
        add_obj(reg, reg.target_obj[ind])

def add_obj(reg, obj):
  tmp = copy.deepcopy(obj.get_strokes())
  new_obj = UnlabeledObject(tmp)
  print("origin", obj.origin_x, obj.origin_y)
  print("origin", new_obj.origin_x, new_obj.origin_y)
  obj.print_strokes()
  eps = 0.001
  tmp = []
  for st in new_obj.get_strokes():
      for pt in st.get_points():
          pt.x = pt.x * eps + (1 - eps) * new_obj.origin_x
          pt.y = pt.y * eps + (1 - eps) * new_obj.origin_y
      tmp.append(Stroke(st.get_points()))
  new_obj = UnlabeledObject(tmp)
  reg.original_obj.append(new_obj)

def main():
    # enter input sample id
    s = 9
    # s = int(input())
    ORG_FILE = 'input_directory/samples/test_samples/a' + str(s) + '.xml'
    TAR_FILE = 'input_directory/samples/test_samples/b' + str(s) + '.xml'
    SAVE_FILE = 'input_directory/samples/test_videos/example9-seq-obj.mp4'
    ORG_FILE = path.join(path.abspath(path.join(__file__, '../../')), ORG_FILE)
    TAR_FILE = path.join(path.abspath(path.join(__file__, '../../')), TAR_FILE)
    SAVE_FILE = path.join(path.abspath(path.join(__file__, '../../')), SAVE_FILE)
    reg = Registration(ORG_FILE, TAR_FILE, mn_stroke_len=3, re_sampling=0.5, flip=False, shift_target_y = 0)

    # add missing objects (temporarily as pre-calculation is running on ssh server)
    add_objects(reg, []) # fill the indices of the missing objects

    st = time.time()
    # p = reg.register(mx_dissimilarity=75)
    # print("Running time:", time.time()-st)
    # print("Optimal registration params:", [np.array(p)])
    p = [
        np.array([ 1.91810036e-01,  1.14645143e+00,  1.33596572e-01,  4.88093049e-02,
       -3.66933320e+01,  1.06142019e+03,  2.20824119e+02]),
       np.array([ 1.28563088e+00,  5.47131699e-01,  1.32297177e-01,  2.72717307e-04,
       -1.81945273e-09,  9.95375839e+02, -7.48717701e+00]),
        np.array([ 1.26276361e+00,  9.43611338e-01, -2.06393644e-01,  1.02706147e+00,
       -3.97088657e-01,  1.38495550e+03, -1.12055765e+02]),
       np.array([ 6.29412479e-01,  8.14474661e-01,  2.91879526e-05,  6.94840005e-02,
        9.61637390e-02,  9.39366106e+02, -1.19839475e+02]), 
       np.array([ 4.86998150e-01,  8.40672581e-01, -1.49778653e-10, -1.19399221e-01,
        3.90920179e-02,  7.83428959e+02,  2.58489434e+02]),
    #    np.array([ 1.31471719e+00,  8.90295585e-01, -7.54362306e-02,  6.50186027e-01,
    #    -4.35733611e-01,  3.85115604e+02, -1.21612762e+02]), 
    #    np.array([ 6.25517185e-01,  8.08183487e-01,  1.23138564e-01, -6.31528960e-02,
    #    -2.30786642e-06, -6.15874738e+01, -1.22314337e+02]), 
    #    np.array([ 4.80301654e-01,  8.32812773e-01,  3.99573817e-02, -7.01261326e-02,
    #    -4.46017181e-05, -2.15891777e+02,  2.59053341e+02])
    ]
    x, y = p[0][5], p[0][6]
    for i in range(len(p)):
        p[i][5] -= x
        p[i][6] -= y

    t = []
    for lst in p:
        t.append(RegistrationUtils.obtain_transformation_matrix(lst))
    print("Optimal matrix transformation params:", t)

    anim = SketchAnimation(reg.original_obj, reg.target_obj)
    anim.seq_animate_all(p, save=True
                            , file=SAVE_FILE)
                    
if __name__ == "__main__":
    main()