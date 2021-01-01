
import numpy as np
from matplotlib import pyplot as plt

# test custom loss function 

def np_knn_loss(sketches, p, maxlen=10):
        # sketches = tf.identity(sketches)
        # sketches (batch, 2, 126, 3, 1)  p(batch, 7)
        org = sketches[:, 0, :, :2, 0]
        tar = sketches[:, 1, :, :2, 0]
        # org: (batch, 126, 2)  tar: (batch, 126, 2)
        org_pen = sketches[:, 0, :, 2, 0]
        tar_pen = sketches[:, 1, :, 2, 0]
        # org_pen: (batch, 126) tar_pen: (batch, 126)   represents the pen state in stroke-3 format

        # obtain transformation matrix parameters
        t = []
        t.append(p[:, 0] * (np.cos(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) - p[:, 4] * np.sin(p[:, 2])))
        t.append(p[:, 0] * (p[:, 3] * np.cos(p[:, 2]) - np.sin(p[:, 2])))
        t.append(p[:, 5])
        t.append(p[:, 1] * (np.sin(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) + p[:, 4] * np.cos(p[:, 2])))
        t.append(p[:, 1] * (p[:, 3] * np.sin(p[:, 2]) + np.cos(p[:, 2])))
        t.append(p[:, 6])
        t = np.expand_dims(t, -1)
        # t: (batch, 6, 1)

        # apply transformation on all points in original 
        org_x = org[:, :, 0] * t[0] + org[:, :, 1] * t[1] + t[2]
        org_y = org[:, :, 0] * t[3] + org[:, :, 1] * t[4] + t[5]

        # org_x: represent x coords (batch, 126)
        # org_y: represent x coords (batch, 126)
        org_x = np.expand_dims(org_x, 2)
        org_y = np.expand_dims(org_y, 2)
        # org_x: (represent x coords) (batch, 126, 1)
        # org_y: (represent x coords) (batch, 126, 1)

        org_cmb = np.concatenate([org_x, org_y], axis=-1)
        # org_cmb : (batch, 126, 2)

        org_cmb = np.expand_dims(org_cmb, 1)
        # org_cmb: (batch, 1, 126, 2)

        tar_cmb = np.expand_dims(tar, 2)
        # tar_cmb: (batch, 126, 1, 2)

        # obtain pairwise differences between original and target sketches
        diff = org_cmb - tar_cmb
        # diff: (batch, 126, 126, 2)
 
        sm = np.sum(diff ** 2, axis=-1)
        sm_sqrt = np.sqrt(sm)
        # sm_sqrt: (batch, 126, 126)

        # obtain nearest points from org->tar + from tar->org
        mn = np.min(sm_sqrt, axis=-2) * (1 - org_pen) + np.min(sm_sqrt, axis=-1) * (1 - tar_pen)
        # mn: (batch, 126)

        sm_cost = np.sum(mn, axis=1) 
        # sm_cost: (batch, )

        # normalize with the number of points
        sm_cost /= maxlen - np.sum(org_pen, axis=-1) + maxlen - np.sum(tar_pen, axis=-1) 

        return sm_cost         

# ## translation test
# sketches = []
# sketches.append([[[1], [1], [0]],[[2], [2], [0]],[[3], [3], [0]]])
# sketches.append([[[0], [0], [0]], [[1], [1], [0]], [[2], [2], [0]]])
# p = [[1, 1, 0, 0, 0, 1, 1]]
# print(knn_loss(np.float32(np.asarray([sketches])), np.float32(np.asarray(p))))
# # worked

## scaling test
def _pad_sketches(sketches, maxlen=10, inf=1e9):
        converted_sketches = []
        for i in range(len(sketches)):
                tmp = []
                if len(sketches[i]) >= maxlen:
                        tmp = np.array(sketches[i][:maxlen-1])
                else:
                        tmp = sketches[i]
                # add at least one padding
                extra = np.repeat(np.array([[inf, inf, 1]]), maxlen-len(tmp), axis=0)
                converted_sketches.append(np.concatenate((tmp, extra), axis=0))
        return np.asarray(converted_sketches)


sketches = []
sketches.append([[[1], [1], [0]],[[2], [2], [1]],[[3], [3], [1]]])
sketches.append([[[0], [0], [0]], [[1], [1], [0]], [[2000], [2000], [1]]])
p = [[1, 1, 0, 0, 0, 0, 0]]
sketches = np.squeeze(sketches, axis=-1)
sketches = _pad_sketches(sketches)
sketches = np.expand_dims(sketches, axis=-1)
print(np_knn_loss(np.float32(np.asarray([sketches])), np.float32(np.asarray(p))))
## worked

# ## rotation test
# sketches = []
# sketches.append([[[1], [1], [0]],[[2], [2], [0]],[[3], [3], [0]]])
# sketches.append([[[0], [0], [0]], [[1], [1], [0]], [[2], [2], [0]]])
# p = [[1, 1, np.pi, 0, 0, 0, 0]]
# print(knn_loss(np.float32(np.asarray([sketches])), np.float32(np.asarray(p))))
# ## worked

B = [[-2.46913580e+00, -2.46913580e+00],
        [-7.40740741e+00,  0.00000000e+00],
        [-1.48148148e+01,  9.87654321e+00],
        [-2.22222222e+01,  2.46913580e+01],
        [-2.96296296e+01,  4.69135802e+01],
        [-4.19753086e+01,  8.64197531e+01],
        [-4.69135802e+01,  1.13580247e+02],
        [-4.93827160e+01,  1.35802469e+02],
        [-5.18518519e+01,  1.50617284e+02],
        [-5.18518519e+01,  1.55555556e+02],
        [-4.69135802e+01,  1.50617284e+02],
        [-4.44444444e+01,  1.35802469e+02],
        [-4.19753086e+01,  1.11111111e+02],
        [-3.70370370e+01,  7.90123457e+01],
        [-2.96296296e+01,  4.44444444e+01],
        [-1.72839506e+01,  2.46913580e+00],
        [-7.40740741e+00, -1.97530864e+01],
        [ 1.23456790e+01, -3.95061728e+01],
        [ 3.20987654e+01, -2.22222222e+01],
        [ 4.44444444e+01,  1.97530864e+01],
        [ 5.92592593e+01,  8.14814815e+01],
        [ 7.16049383e+01,  1.23456790e+02],
        [ 8.14814815e+01,  1.55555556e+02],
        [ 8.14814815e+01,  1.60493827e+02],
        [ 5.67901235e+01,  1.60493827e+02],
        [ 3.45679012e+01,  1.60493827e+02],
        [-2.96296296e+01, 1.58024691e+02],
        [-4.93827160e+01, 1.58024691e+02],
        [-7.65432099e+01,  1.55555556e+02],
        [-7.90123457e+01,  1.53086420e+02]]


A = [[ 1.20614996e+00,  5.06497028e-01],
        [ 6.47277281e+00,  2.73325515e+00],
        [ 1.57936605e+01,  1.19363399e+01],
        [ 2.62513109e+01,  2.58297828e+01],
        [ 3.84141053e+01,  4.67587628e+01],
        [ 5.92538107e+01  ,8.39855751e+01],
        [ 7.02042472e+01  ,1.09664124e+02],
        [ 7.76688001e+01 , 1.30711525e+02],
        [ 8.34282090e+01 , 1.44723388e+02],
        [ 8.45649717e+01 , 1.49413746e+02],
        [ 7.87299675e+01 , 1.44841809e+02],
        [ 7.29705586e+01 , 1.30829945e+02],
        [ 6.49376243e+01 , 1.07437366e+02],
        [ 5.28504252e+01 , 7.70684589e+01],
        [ 3.78457239e+01 , 4.44135837e+01],
        [ 1.64376372e+01 , 4.84159236e+00],
        [ 1.92572196e+00 ,-1.60281772e+01],
        [-2.14142949e+01 ,-3.43159258e+01],
        [-3.62285913e+01 ,-1.74259889e+01],
        [-3.83117120e+01 , 2.27381071e+01],
        [-3.81969026e+01,  8.17228458e+01],
        [-4.02800232e+01,  1.21886942e+02],
        [-4.22875486e+01,  1.52611111e+02],
        [-4.11507859e+01,  1.57301469e+02],
        [-1.76595784e+01,  1.56709365e+02],
        [ 3.48250830e+00,  1.56176471e+02],
        [ 6.39912663e+01,  1.52291820e+02],
        [ 8.27842323e+01,  1.51818136e+02],
        [ 1.08056179e+02,  1.48821642e+02],
        [ 1.09836919e+02,  1.46417252e+02]]
A = np.array(A)
B = np.array(B)
sum = 0
for p1 in A:
        tmp = 1e9
        k = None
        for p2 in B:
                if np.sqrt((p2[0]-p1[0])**2 + (p2[0]-p1[0])**2)< tmp:
                        tmp = np.sqrt((p2[0]-p1[0])**2 + (p2[0]-p1[0])**2)
                        k = p2
        sum += tmp
        print(p1)
        print(k)
        print(tmp)
        print()

plt.plot(A[:,0], A[:,1], 'b', '-.')
plt.plot(B[:,0], B[:,1], 'r', '-.')
plt.show()
print(sum/(len(A) + len(B)))
