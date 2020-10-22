from utils.ObjectUtil import ObjectUtil
import numpy as np
from scipy.optimize import minimize, basinhopping
from utils.NearestSearch import NearestSearch
from lapjv import lapjv
from utils.RegistrationUtils import RegistrationUtils, RegisterTwoObjects
from sketch_object.UnlabeledObject import UnlabeledObject
import copy
from multiprocessing import Pool
import multiprocessing
from sketch_object.Stroke import Stroke
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
    # original_strokes_collection = [[0, 1, 2, 6], [3, 4, 5], [7, 8, 9], [10], [11], [12], [13], [14]] 
    # target_strokes_collection = [[0], [1], [2], [3], [4], [5]]

    def __init__(self, org_file, tar_file, re_sampling=1.0, mn_stroke_len=0, flip=False, shift_target_x = 0.0, shift_target_y = 0.0,
                 shearing_cost=RegistrationUtils._shearing_cost, translation_cost=RegistrationUtils._translation_cost,
                 rotation_cost=RegistrationUtils._rotation_cost, scaling_cost=RegistrationUtils._scaling_cost):

        self.sh_cost, self.tr_cost, self.ro_cost, self.sc_cost = shearing_cost, translation_cost, rotation_cost, scaling_cost

        self.original_obj, self.origninal_labels = ObjectUtil.xml_to_UnlabeledObjects(org_file,
                                                               re_sampling=re_sampling, mn_len=mn_stroke_len, flip=flip)
        self.target_obj, self.target_labels = ObjectUtil.xml_to_UnlabeledObjects(tar_file,
                                                             re_sampling=re_sampling, mn_len=mn_stroke_len, flip=flip, shift_y=shift_target_y, shift_x=shift_target_x)
        self.core_cnt = multiprocessing.cpu_count()
        print("CPU count:", self.core_cnt)
        print("Original sketch labels", self.origninal_labels)
        print("Target sketch labels", self.target_labels)
    
    def register(self, mx_dissimilarity = 50):
        n, m = len(self.original_obj), len(self.target_obj)
        dim = max(n,m)
        self.res_matrix = np.zeros((dim, dim))
        self.tra_matrix = np.zeros((dim, dim, 7))            

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
                self.res_matrix[i, j] = d
                self.tra_matrix[i, j] = p
        
        print("res_matrix", self.res_matrix)

        # calculate the minimum assignment
        org_asg, tar_asg, total_cost = lapjv(self.res_matrix)
        print("selection", org_asg)
        final_transformation = np.zeros((n, 7))
        added_objects = []

        for i, ind in enumerate(org_asg):
            dissimilarity = self.res_matrix[i, ind]
            if i < n and ind < m:
                ln = max(len(self.original_obj[i]), len(self.target_obj[ind]))
                ref_obj = ObjectUtil.object_restructure(self.original_obj[i], n=ln)
                tar_obj = ObjectUtil.object_restructure(self.target_obj[ind], n=ln)
                dissimilarity = RegistrationUtils.calc_dissimilarity(ref_obj, tar_obj, self.tra_matrix[i, ind], cum_ang=True, turning_ang=False)
            print(dissimilarity, self.res_matrix[i, ind])
            # check if one of the objects is dummy or their dissimilarity is above the maximum threshold
            if dissimilarity > mx_dissimilarity:
                diff = dissimilarity != RegistrationUtils.inf
                # handle the case when n > m by making the object vanish into its origin:
                if n > m or diff:
                    self.tra_matrix[i, ind] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.original_obj[i].origin_x, self.original_obj[i].origin_y])        
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
                final_transformation[i] = self.tra_matrix[i, ind]
        print("added_objects:", added_objects)
        return final_transformation

        """redistribute the assignment of the identical object by considering their spatial relation 
        
        Parameters:
            target_groups: the indecies of orginal object grouped
            org_asg: the initial assignment of the orginal objects
            tar_asg: the initial assignment of the target objects

        Returns:
            None. mutate org_asg and tar_asg
        """     
    def spatial_redistribution(self, target_groups, org_asg, tar_asg):
        for group in target_groups:
            n = len(group)
            if n == 1:
                continue
            org = []
            for obj_ind in group:
                org.append(tar_asg[obj_ind])
            weight_matrix = np.zeros((n, n))
            for i in org:
                for j in group:
                    weight_matrix[i][j] = self.tra_matrix[i][j][5] ** 2 + self.tra_matrix[i][j][6] ** 2
            row_ind, col_ind, _ = lapjv(weight_matrix)
            for i, ind in enumerate(row_ind):
                org_asg[org[i]] = group[ind]
            for i, ind in enumerate(col_ind):
                tar_asg[group[i]] = org[ind]

            


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