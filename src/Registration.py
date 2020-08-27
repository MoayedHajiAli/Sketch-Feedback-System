from ObjectUtil import ObjectUtil
import numpy as np
from scipy.optimize import minimize, basinhopping
from Nearest_search import Nearest_search
from lapjv import lapjv
from RegistrationUtils import RegistrationUtils, RegisterTwoObjects
from UnlabeledObject import UnlabeledObject
import copy
from multiprocessing import Pool
import multiprocessing
from Stroke import Stroke
import sys

class Registration:

    # matplotlib default color cycle
    # blue -> orange -> green -> red -> purple -> brown -> pink -> gray -> yellow -> light-blue
    # manual strokes collections for a2 -> b2

    #example3
    # original_strokes_collection = [[0], [1], [2, 3], [4, 5, 6], [7, 8], [9, 10, 11, 12]]
    # target_strokes_collection = [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10, 11, 12]]

    # example4
    # original_strokes_collection = [[0], [1], [2, 3], [4, 5], [6], [7], [8]]
    # target_strokes_collection = [[0], [1], [2], [3, 4], [5, 6], [7], [8, 9, 10]]

    # # # example5
    # original_strokes_collection = [[0], [1], [2, 3, 4, 5], [6], [7, 8, 9, 10]]
    # target_strokes_collection = [[0], [1], [2, 3], [4, 5, 6], [7, 8]]

    # example6
    # original_strokes_collection = [[0], [1], [2], [3, 4], [5, 6], [7], [8], [9, 10], [11], [12], [13, 14, 15], [16]]
    # target_strokes_collection = [[0], [1], [2], [3, 4], [5, 6], [7], [8], [9, 10], [11], [12, 13, 14, 15], [16], [17, 18]]
    
    # example7
    original_strokes_collection = [[0, 1, 2, 6], [3, 4, 5], [7, 8, 9], [10], [11], [12], [13], [14]] 
    target_strokes_collection = [[0], [1], [2], [3], [4], [5]]

    def __init__(self, org_file, tar_file, re_sampling=1.0, mn_stroke_len=0, flip=False, shift_target_x = 0.0, shift_target_y = 0.0,
                 shearing_cost=RegistrationUtils._shearing_cost, translation_cost=RegistrationUtils._translation_cost,
                 rotation_cost=RegistrationUtils._rotation_cost, scaling_cost=RegistrationUtils._scaling_cost):

        self.sh_cost, self.tr_cost, self.ro_cost, self.sc_cost = shearing_cost, translation_cost, rotation_cost, scaling_cost

        self.original_obj = ObjectUtil.xml_to_UnlabeledObjects(org_file, self.original_strokes_collection,
                                                               re_sampling=re_sampling, mn_len=mn_stroke_len, flip=flip)
        self.target_obj = ObjectUtil.xml_to_UnlabeledObjects(tar_file, self.target_strokes_collection,
                                                             re_sampling=re_sampling, mn_len=mn_stroke_len, flip=flip, shift_y=shift_target_y, shift_x=shift_target_x)
        self.core_cnt = multiprocessing.cpu_count()
    
    def register(self, mx_dissimilarity = 50):
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
                    d, p = RegistrationUtils.inf, np.zeros(7)
                else:
                    d, p = res[t]
                    t += 1
                res_matrix[i, j] = d
                tra_matrix[i, j] = p
        
        print("res_matrix", res_matrix)

        # calculate the minimum assignment
        row_ind, col_ind, total_cost = lapjv(res_matrix)
        print("selection", row_ind)
        final_transformation = np.zeros((n, 7))
        added_objects = []

        for i, ind in enumerate(row_ind):
            dissimilarity = res_matrix[i, ind]
            if i < n and ind < m:
                ln = max(len(self.original_obj[i]), len(self.target_obj[ind]))
                ref_obj = ObjectUtil.object_restructure(self.original_obj[i], n=ln)
                tar_obj = ObjectUtil.object_restructure(self.target_obj[ind], n=ln)
                dissimilarity = RegistrationUtils.calc_dissimilarity(ref_obj, tar_obj, tra_matrix[i, ind], cum_ang=True, turning_ang=False)
            print(dissimilarity, res_matrix[i, ind])
            # check if one of the objects is dummy or their dissimilarity is above the maximum threshold
            if dissimilarity > mx_dissimilarity:
                diff = dissimilarity != RegistrationUtils.inf
                # handle the case when n > m by making the object vanish into its origin:
                if n > m or diff:
                    tra_matrix[i, ind] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.original_obj[i].origin_x, self.original_obj[i].origin_y])        
                # handle the case when m > n by creating a new object in the orginal sketch, identical to the target sketch but scalled by a very small scale
                if m > n or diff:
                    tmp = copy.deepcopy(self.target_obj[ind].get_strokes())
                    new_obj = UnlabeledObject(tmp)
                    eps = 0.001
                    tmp = []
                    for st in new_obj.get_strokes():
                        for pt in st.get_points():
                            pt.x = pt.x * eps + (1 - eps) * new_obj.origin_x
                            pt.y = pt.y * eps + (1 - eps) * new_obj.origin_y
                        tmp.append(Stroke(st.get_points()))
                    new_obj = UnlabeledObject(tmp)
                    print(final_transformation.shape)
                    final_transformation = np.append(final_transformation, np.array([[1/eps, 1/eps, 0.0, 0.0, 0.0, 0.0, 0.0]]), axis=0)
                    print(final_transformation.shape)
                    self.original_obj.append(new_obj)
                    added_objects.append(ind)

            if i < n:
                final_transformation[i] = tra_matrix[i, ind]
        print("added_objects:", added_objects)
        return final_transformation

    # wrapper function for calling optimize on a RegisterTwoObjects
    def _optimize(self, reg):
        x_dif = reg.tar_obj.origin_x - reg.ref_obj.origin_x
        y_dif = reg.tar_obj.origin_y - reg.ref_obj.origin_y
        p = np.array([1.0, 1.0, 0.0, 0.0, 0.0, x_dif, y_dif])
        return reg.optimize(p = p, params=True)


    # obtain total transformation **parameters** cost
    def total_cost(self, p, mn_x, mx_x, mn_y, mx_y, ln):
        tot = 0.0
        tot += self.sc_cost(p[0], p[1], ln)
        tot += self.ro_cost(p[2], ln)
        tot += self.sh_cost(p[3], p[4], mn_x, mn_y, mx_x, mx_y, ln)
        tot += self.tr_cost(p[5], p[6], ln)
        return tot

if __name__ == "__main__":
    pass