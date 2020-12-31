import sys
sys.path.insert(0, '../')

from register.Registration import Registration
from animator.SketchAnimation import SketchAnimation
import matplotlib.pyplot as plt

# register two sketches
reg = Registration('../input_directory/samples/test_samples/a' + str(s) + '.xml', './input_directory/samples/test_samples/b' + str(s) + '.xml', mn_stroke_len=3, re_sampling=0.5, flip=False, shift_target_y = 0)

# add missing objects (temporarily as pre-calculation is running on ssh server)
add_objects(reg, []) # fill the indices of the missing objects

st = time.time()
p = reg.register(mx_dissimilarity=100)
print("Running time:", time.time()-st)
print("Optimal registration params:", [np.array(p)])

t = []
for lst in p:
    t.append(RegistrationUtils.obtain_transformation_matrix(lst))
print("Optimal matrix transformation params:", t)

SketchAnimation = SketchAnimation(reg.original_obj, reg.target_obj)
SketchAnimation.seq_animate_all(p, save=False
                        , file="./test_videos/example6-seq.mp4")