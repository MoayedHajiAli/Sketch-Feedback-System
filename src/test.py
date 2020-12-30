
import numpy as np

# test custom loss function 

def np_knn_loss(sketches, p):
        # sketches (batch, 2, 126, 3, 1)  p(batch, 7)
        org = sketches[:, 0, :, :2, 0]
        tar = sketches[:, 1, :, :2, 0]
        org_pen = sketches[:, 0, :, 2, 0]
        tar_pen = sketches[:, 1, :, 2, 0]
        # org: (batch, 126, 2)  tar: (batch, 126, 2)
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
        org_cmb = np.expand_dims(org_cmb, 1)
        # org_cmb: (batch, 1, 126, 2)

        tar_cmb = np.expand_dims(tar, 2)
        # tar_cmb: (batch, 126, 1, 2)

        # obtain pairwise differences between original and target sketches
        diff = org_cmb - tar_cmb
        # diff: (batch, 126, 126, 2)
        print(diff[0])
        sm = np.sum(diff ** 2, axis=-1)
        print(sm)
        sm_sqrt = np.sqrt(sm)
        # print(sm)
        # sm_sqrt: (batch, 126, 126)

        # obtain nearest points from org->tar + from tar->org
        mn = np.min(sm, axis=-2) * (1 - org_pen) + np.min(sm, axis=-1) * (1 - tar_pen)
        print(mn)
        # mn: (batch, 126)

        sm_cost = np.sum(mn, axis=1) 
        # sm_cost: (batch, )

        # normalize with the number of points
        # sm_cost /= 126 - np.sum(org_pen, axis=-1) + 126 - np.sum(tar_pen, axis=-1) 

        return sm_cost         

# ## translation test
# sketches = []
# sketches.append([[[1], [1], [0]],[[2], [2], [0]],[[3], [3], [0]]])
# sketches.append([[[0], [0], [0]], [[1], [1], [0]], [[2], [2], [0]]])
# p = [[1, 1, 0, 0, 0, 1, 1]]
# print(knn_loss(np.float32(np.asarray([sketches])), np.float32(np.asarray(p))))
# # worked

## scaling test
def _pad_sketches(sketches, maxlen=5, inf=1e5):
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
sketches.append([[[1], [1], [0]],[[2], [2], [0]],[[3], [3], [0]]])
sketches.append([[[0], [0], [0]], [[1], [1], [0]], [[2], [2], [0]]])
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
