from ObjectUtil import ObjectUtil
from autograd import grad
import numpy as np
from scipy.optimize import minimize, basinhopping
from Nearest_search import Nearest_search
from lapjv import lapjv
from RegisterationUtils import RegsiterationUtils
from UnlabeledObject import UnlabeledObject
import copy
from multiprocessing import Pool
import multiprocessing

class Registration:

    # matplotlib default color cycle
    # blue -> orange -> green -> red -> purple -> brown -> pink -> gray -> yellow -> light-blue
    # manual strokes collections for a2 -> b2

    #example3
    # original_strokes_collection = [[0], [1], [2, 3], [4, 5, 6], [7, 8], [9, 10, 11, 12]]
    # target_strokes_collection = [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10, 11, 12]]

    # example4
    # original_strokes_collection = [[0], [1], [2, 3], [4, 5], [6], [7], [8]]
    target_strokes_collection = [[0], [1], [2], [3, 4], [5, 6], [7], [8, 9, 10]]

    # # # example5
    original_strokes_collection = [[0], [1], [2, 3, 4, 5], [6], [7, 8, 9, 10]]
    # target_strokes_collection = [[0], [1], [2, 3], [4, 5, 6], [7, 8]]

    # example6
    # original_strokes_collection = [[0], [1], [2], [3, 4], [5, 6], [7], [8], [9, 10], [11], [12], [13, 14, 15], [16]]
    # target_strokes_collection = [[0], [1], [2], [3, 4], [5, 6], [7], [8], [9, 10], [11], [12, 13, 14, 15], [16], [17, 18]]

    def __init__(self, org_file, tar_file, re_sampling=1.0, mn_stroke_len=0, flip=False, shift_target_x = 0.0, shift_target_y = 0.0,
                 shearing_cost=RegsiterationUtils._shearing_cost, translation_cost=RegsiterationUtils._translation_cost,
                 rotation_cost=RegsiterationUtils._rotation_cost, scaling_cost=RegsiterationUtils._scaling_cost):

        self.sh_cost, self.tr_cost, self.ro_cost, self.sc_cost = shearing_cost, translation_cost, rotation_cost, scaling_cost

        self.original_obj = ObjectUtil.xml_to_UnlabeledObjects(org_file, self.original_strokes_collection,
                                                               re_sampling=re_sampling, mn_len=mn_stroke_len, flip=flip)
        self.target_obj = ObjectUtil.xml_to_UnlabeledObjects(tar_file, self.target_strokes_collection,
                                                             re_sampling=re_sampling, mn_len=mn_stroke_len, flip=flip, shift_y=shift_target_y, shift_x=shift_target_x)
        self.core_cnt = multiprocessing.cpu_count()
    
    def register(self, mx_dissimilarity = 500):
        n, m = len(self.original_obj), len(self.target_obj)
        dim = max(n,m)
        res_matrix = np.zeros((dim, dim))
        tra_matrix = np.zeros((dim, dim, 7))

        # prepare queue of regiteration objects for multiprocessing
        pro_queue = []
        for obj1 in self.original_obj:
            for obj2 in self.target_obj:
                pro_queue.append(RegisterTwoObjects(obj1, obj2, self.total_cost))
        
        # register all the objects using pooling
        res = []
        with Pool(self.core_cnt) as p:
            res = list(p.map(self._optimize, pro_queue))
        
        # fill the result in the res_matrix
        t = 0
        for i in range(dim):
            print(i)
            # t = np.random.rand(7)
            for j in range(dim):
                if i >= n or j >= m:
                    d, p = RegsiterationUtils.inf, np.zeros(7)
                else:
                    d, p = res[t]
                    t += 1
                res_matrix[i, j] = d
                tra_matrix[i, j] = p
        
        print(res_matrix)

        # calculate the minimum assignment
        row_ind, col_ind, tot = lapjv(res_matrix)
        print(row_ind)
        final_transformation = []
        for i, ind in enumerate(row_ind):
            # check if one of the objects is dummy
            if res_matrix[i, ind] == RegsiterationUtils.inf:
                 # handle the case when n > m by making the object vanish into its origin:
                if n > m:
                    tra_matrix[i, ind] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.original_obj[i].origin_x, self.original_obj[j].origin_y])        
                # handle the case when m > n by creating a new object in the orginal sketch, identical to the target sketch but scalled by a very small scale
                else:
                    tmp = copy.deepcopy(obj.get_strokes())
                    new_obj = UnlabeledObject(tmp)
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
                    tra_matrix[i, ind] = np.array([1/eps, 1/eps, 0.0, 0.0, 0.0, 0.0, 0.0])
                    self.original_obj.append(new_obj)

            final_transformation.append(tra_matrix[i, ind])
        return final_transformation

    # wrapper function for calling optimize on a RegisterTwoObjects
    def _optimize(self, reg):
        x_dif = reg.tar_obj.origin_x - reg.ref_obj.origin_x
        y_dif = reg.tar_obj.origin_y - reg.ref_obj.origin_y
        t = np.array([1.0, 1.0, 0.0, 0.0, 0.0, x_dif, y_dif])
        return reg.optimize(t)

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

    # def find_grad(self):
    #     return grad(self.calc_dissimilarity, argnum=(0))

    def optimize(self, t):
        # current similarity function uses kd-tree, which is not suitable for symbolic automatic differentiation
        # grad = self.find_grad()

        # track function for scipy minimize
        def _track(xk):
            print(xk)

        # transform both object to the origin of the referenced object
        RegsiterationUtils.normalize_coords(self.ref_obj, self.tar_obj, -1)

        self.x1, self.y1 = np.array(self.tar_obj.get_x()), np.array(self.tar_obj.get_y())
        self.x2, self.y2 = np.array(self.ref_obj.get_x()), np.array(self.ref_obj.get_y())
        self.target_nn = Nearest_search(self.x1, self.y1)

        # calculate min/max coordinates for the referenced object
        self.mn_x, self.mx_x = min(self.x2), max(self.x2)
        self.mn_y, self.mx_y = min(self.y2), max(self.y2)

        # minimize
        minimizer_kwargs = {"method": "BFGS"}
        res = basinhopping(self.calc_dissimilarity, t, minimizer_kwargs=minimizer_kwargs, disp=True, niter=1)
        d, p = res.fun, res.x

        # restore the origin by transforming both object to the (0,0) coords
        RegsiterationUtils.normalize_coords(self.ref_obj, self.tar_obj, 1)

        return d, p

if __name__ == '__main__':
    pass