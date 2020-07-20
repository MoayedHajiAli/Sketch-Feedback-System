from ObjectUtil import ObjectUtil
import autograd.numpy as np
from autograd import grad
import numpy as npn
from scipy.optimize import minimize, basinhopping
from Nearest_search import Nearest_search
from lapjv import lapjv
from RegisterationUtils import RegsiterationUtils
from UnlabeledObject import UnlabeledObject


class Registration:

    # matplotlib default color cycle
    # blue -> orange -> green -> red -> purple -> brown -> pink -> gray -> yellow -> light-blue
    # manual strokes collections for a2 -> b2

    # #example3
    # original_strokes_collection = [[0], [1], [2, 3], [4, 5, 6], [7, 8], [9, 10, 11, 12]]
    # target_strokes_collection = [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10, 11, 12]]

    # example4
    original_strokes_collection = [[0], [1], [2, 3], [4, 5], [6], [7], [8]]
    target_strokes_collection = [[0], [1], [2], [3, 4], [5, 6], [7], [8, 9, 10]]

    # # example5
    # original_strokes_collection = [[0], [1], [2, 3, 4, 5], [6], [7, 8, 9, 10]]
    # target_strokes_collection = [[0], [1], [2, 3], [4, 5, 6], [7, 8]]


    def __init__(self, org_file, tar_file, re_sampling=1.0, mn_stroke_len=0, flip=False, shift_target_x = 0.0, shift_target_y = 0.0,
                 shearing_cost=RegsiterationUtils._shearing_cost, translation_cost=RegsiterationUtils._translation_cost,
                 rotation_cost=RegsiterationUtils._rotation_cost, scaling_cost=RegsiterationUtils._scaling_cost):

        self.sh_cost, self.tr_cost, self.ro_cost, self.sc_cost = shearing_cost, translation_cost, rotation_cost, scaling_cost

        self.original_obj = ObjectUtil.xml_to_UnlabeledObjects(org_file, self.original_strokes_collection,
                                                               re_sampling=re_sampling, mn_len=mn_stroke_len, flip=flip)
        self.target_obj = ObjectUtil.xml_to_UnlabeledObjects(tar_file, self.target_strokes_collection,
                                                             re_sampling=re_sampling, mn_len=mn_stroke_len, flip=flip, shift_y=shift_target_y, shift_x=shift_target_x)

    def register(self):
        n, m = len(self.original_obj), len(self.target_obj)
        res_matrix = npn.zeros((n, m))
        tra_matrix = npn.zeros((n, m, 7))
        for i in range(n):
            print(i)
            # t = npn.random.rand(7)
            t = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for j in range(m):
                d, p = RegisterTwoObjects(self.original_obj[i], self.target_obj[j], self.total_cost).optimize(t)
                res_matrix[i, j] = d
                tra_matrix[i, j] = p
        print(res_matrix)

        # calculate the minimum assignment
        row_ind, col_ind, tot = lapjv(res_matrix)
        print(row_ind)
        final_transformation = []
        for i, ind in enumerate(row_ind):
            final_transformation.append(tra_matrix[i, ind])
        return final_transformation

    # obtain total transformation **parameters** cost
    def total_cost(self, p, mn_x, mx_x, mn_y, mx_y, ln):
        tot = 0.0
        tot += self.sc_cost(p[0], p[1], ln)
        tot += self.ro_cost(p[2], ln)
        tot += self.sh_cost(p[3], p[4], mn_x, mn_y, mx_x, mx_y, ln)
        tot += self.tr_cost(p[5], p[6], ln)
        return tot


class RegisterTwoObjects:
    # obj2 is the referenced object and obj1 is the target object
    def __init__(self, ref_obj:UnlabeledObject, tar_obj:UnlabeledObject, cost_fun):
        self.tar_obj = tar_obj
        self.ref_obj = ref_obj
        self.total_cost = cost_fun
        # transform both object to the origin of the referenced object
        RegsiterationUtils.normalize_coords(self.ref_obj, self.tar_obj, -1)

        self.x1, self.y1 = np.array(tar_obj.get_x()), np.array(tar_obj.get_y())
        self.x2, self.y2 = np.array(ref_obj.get_x()), np.array(ref_obj.get_y())
        self.target_nn = Nearest_search(self.x1, self.y1)

        # calculate min/max coordinates for the referenced object
        self.mn_x, self.mx_x = min(self.x2), max(self.x2)
        self.mn_y, self.mx_y = min(self.y2), max(self.y2)

    # dissimilarity function for the two objects of the class
    # p has 7 parameters:
        # p[0]: the scaling the x direction
        # p[1]: the scaling the y direction
        # p[2]: rotation for theta degrees (counter clock-wise in radian)
        # p[3]: shearing in the x axis
        # p[4]: shearing in the y axis
        # p[5]: translation in the x direction
        # p[6]: translation in the y direction
    def calc_dissimilarity(self, p):
        x = []
        y = []
        t = RegsiterationUtils.obtain_transformation_matrix(p)

        for i in range(len(self.x2)):
            x.append(t[0] * self.x2[i] + t[1] * self.y2[i] + t[2])
            y.append(t[3] * self.x2[i] + t[4] * self.y2[i] + t[5])


        # the following block is dedicated to take into account
        # the turning angle, length, and distance. I am commenting them until I perform separated tests on them.

        # cum1 = self.calc_turning(x[0] - 1, y[0], x[0], y[0], x[1], y[1]) * a11 / a11
        # cum2 = self.calc_turning(self.x1[0] - 1, self.y1[0], self.x1[0], self.y1[0], self.x1[1],
        #                          self.y1[1]) * a11 / a11
        #
        # ang = 360.0 - (cum2 - cum1)
        # if ang > cum2 - cum1:
        #     ang = cum2 - cum1
        # tot = ang

        # tot = (ang/180.0) ** 2
        # tot = ((x[0] - self.x1[0])**2 + (y[0] - self.y1[0])**2) * 0.01
        # i, j represent the current points target, referenced stroke respectively
        # i = j = 0
        # for _ in range(len(self.obj1)):
        #
        #     if j + 2 < len(self.obj1):
        #         t1 = self.calc_turning(x[j], y[j], x[j + 1], y[j + 1], x[j + 2], y[j + 2]) * a11 / a11
        #         t2 = self.calc_turning(self.x1[i], self.y1[i], self.x1[i + 1], self.y1[i + 1], self.x1[i + 2],
        #                                self.y1[i + 2]) * a11 / a11
        #
        #         ang = 360.0 - (t2 - t1)
        #         if ang > t2 - t1:
        #             ang = t2 - t1
        #         # tot += (ang/180.0) ** 2
        #
        #         cum1 += t1
        #         cum2 += t2
        #         ang = 360.0 - (cum2 - cum1)
        #         if ang > cum2 - cum1:
        #             ang = cum2 - cum1
        #         #tot += (ang / 180.0) ** 2
        #
        #     if i + 1 < len(self.obj1):
        #         ln1 = np.sqrt((x[j + 1] - x[j]) ** 2 + (y[j + 1] - y[j]) ** 2)
        #         ln2 = np.sqrt((self.x1[i + 1] - self.x1[j]) ** 2 + (self.y1[i + 1] - self.y1[i]) ** 2)
        #         #tot += (ln2 - ln1) ** 2
        #
        #     # # find nearest point from the referenced stroke to the ith point
        #     # # of the target stroke
        #     # mn = 1000000000
        #     # for k in range(len(x)):
        #     #     mn = min(mn, (x[k] - self.x1[i]) ** 2 + (y[k] - self.y1[i]) ** 2)
        #     # tot += mn
        #     #
        #     # # find nearest point from the target stroke to the jth point
        #     # # of the referenced stroke
        #     # mn = 1000000000
        #     # for k in range(len(x)):
        #     #     mn = min(mn, (x[j] - self.x1[k]) ** 2 + (y[j] - self.y1[k]) ** 2)
        #     # tot += mn
        #
        #     # print(x[0], self.x1[0])
        #     # tot += ((x[j] - self.x1[i])**2 + (y[j] - self.y1[i])**2) * 0.01
        #
        #     i = i + 1
        #     j = j + 1

        tot = 0.0

        x = list(map(lambda q: q if isinstance(q, np.float64) else q._value, x))
        y = list(map(lambda q: q if isinstance(q, np.float64) else q._value, y))
        reference_nn = Nearest_search(np.array(x), np.array(y))

        # find nearest point from the target stroke to the points
        # of the referenced stroke
        tot += self.target_nn.query(x, y)
        # print("tot1", tot)
        # find nearest point from the referenced stroke to the ith point
        # # of the target stroke
        tot += reference_nn.query(self.x1, self.y1)
        # print("tot2", tot)
        tran_cost = self.total_cost(p, self.mn_x, self.mx_x, self.mn_y, self.mx_y, len(x))
        # print("tran cost", tran_cost/len(x))
        return (tot + tran_cost)/(len(x) + len(self.x1))

    def find_grad(self):
        return grad(self.calc_dissimilarity, argnum=(0))

    def optimize(self, t=npn.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        # current similarity function uses kd-tree, which is not suitable for symbolic automatic differentiation
        # grad = self.find_grad()

        # track function for scipy minimize
        def _track(xk):
            print(xk)

        minimizer_kwargs = {"method": "BFGS"}
        res = basinhopping(self.calc_dissimilarity, t, minimizer_kwargs=minimizer_kwargs, disp=True, niter=1)
        # res = basinhopping(self.calc_dissimilarity, t, method="BFGS", options={'gtol': 1e-6})
        d, p = res.fun, res.x

        # restore the origin
        # transform both object to the origin of the referenced object
        RegsiterationUtils.normalize_coords(self.ref_obj, self.tar_obj, 1)

        return d, p

