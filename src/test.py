
import numpy as np

# test custom loss function 

def knn_loss(snpetches, p):
        org = snpetches[:, 0, :, :2, 0]
        tar = snpetches[:, 1, :, :2, 0]
        # # obtain transformation matrix parameters
        t = []
        t.append(p[:, 0] * (np.cos(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) - p[:, 4] * np.sin(p[:, 2])))
        t.append(p[:, 0] * (p[:, 3] * np.cos(p[:, 2]) - np.sin(p[:, 2])))
        t.append(p[:, 5])
        t.append(p[:, 1] * (np.sin(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) + p[:, 4] * np.cos(p[:, 2])))
        t.append(p[:, 1] * (p[:, 3] * np.sin(p[:, 2]) + np.cos(p[:, 2])))
        t.append(p[:, 6])
        t = np.expand_dims(t, -1)
        # apply transformation on all points in original 
        # org (None, 128, 3) , p(None,)
        org1 = org[:, :, 0] * t[0]
        org1 = org[:, :, 0] * t[0] + org[:, :, 1] * t[1] + t[2]
        org2 = org[:, :, 0] * t[3] + org[:, :, 1] * t[4] + t[5]
        # org1, org2 = org[:, :, 0] * p[:,0] + org[:, :, 1] * p[:,1] + p[:,2],  org[:, :, 0] * p[:,3] + org[:, :, 1] * p[:,4] + p[:,5]
        org1 = np.expand_dims(org1, 2)
        org2 = np.expand_dims(org2, 2)
        org_cmb = np.concatenate([org1, org2], axis=-1)
        org_cmb = np.expand_dims(org_cmb, 1)
        tar_cmb = np.expand_dims(tar, 2)
        print("org", np.array(org_cmb))
        print("tar", np.array(tar_cmb))
        print(np.array(org_cmb).shape)
        print(np.array(tar_cmb).shape)
        dif = org_cmb - tar_cmb
        print("dif", dif)
        sm = np.sum(dif ** 2, axis=-1)
        sm = np.sqrt(sm)
        print("sum", sm)
        mn = np.min(sm, axis=-2)
        return np.sum(mn, axis=1)  

# ## translation test
# sketches = []
# sketches.append([[[1], [1], [0]],[[2], [2], [0]],[[3], [3], [0]]])
# sketches.append([[[0], [0], [0]], [[1], [1], [0]], [[2], [2], [0]]])
# p = [[1, 1, 0, 0, 0, 1, 1]]
# print(knn_loss(np.float32(np.asarray([sketches])), np.float32(np.asarray(p))))
# # worked

## scaling test
sketches = []
sketches.append([[[1], [1], [0]],[[2], [2], [0]],[[3], [3], [0]]])
sketches.append([[[0], [0], [0]], [[1], [1], [0]], [[2], [2], [0]]])
p = [[2, 2, 0, 0, 0, 0, 0]]
print(knn_loss(np.float32(np.asarray([sketches])), np.float32(np.asarray(p))))
## worked

# ## rotation test
# sketches = []
# sketches.append([[[1], [1], [0]],[[2], [2], [0]],[[3], [3], [0]]])
# sketches.append([[[0], [0], [0]], [[1], [1], [0]], [[2], [2], [0]]])
# p = [[1, 1, np.pi, 0, 0, 0, 0]]
# print(knn_loss(np.float32(np.asarray([sketches])), np.float32(np.asarray(p))))
# ## worked
