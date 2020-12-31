import sys
sys.path.insert(0, '../')

from register.Registration import RegisterTwoObjects
from animator.SketchAnimation import SketchAnimation
import matplotlib.pyplot as plt

# enter input sample id
s = int(input())
reg = Registration('../input_directory/samples/test_samples/a' + str(s) + '.xml', '../input_directory/samples/test_samples/b' + str(s) + '.xml', mn_stroke_len=3, re_sampling=0.5, flip=False, shift_target_y = 0)

# enter the indices of the original and target object
a, b = map(int, input().split())

st = time.time()
obj1, obj2 = reg.original_obj[i], reg.target_obj[j]
print("Orignal object length:", len(obj1))
print("Target object length:", len(obj2))

d, p = RegisterTwoObjects(obj1, obj2, reg.total_cost).optimize()
print("optimal cost:", d)
print("optimal transforamtion params:", [np.array(p)])
print("Running time: ", time.time()-st)

animation = SketchAnimation([obj1], [obj2])
animation.seq_animate_all([p], save=False, file="../test_videos/example7-obj3-4.mp4")
plt.show()