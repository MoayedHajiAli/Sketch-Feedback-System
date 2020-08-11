import numpy as np
from UnlabeledObject import UnlabeledObject
import copy
from Nearest_search import Nearest_search

class RegistrationUtils:

    inf = 1e9+7
    # default shearing cost function where
        # a is the shearing parallel to the x axis
        # b is the shearing parallel to the y axis
    @staticmethod
    def _shearing_cost(a, b, mn_x, mn_y, mx_x, mx_y, ln, fac_x=10, fac_y=10):
        a = abs(a)
        b = abs(b)

        cost = ln * (fac_x * a + fac_y * b)
        #cost = ln * (fac_x * ((a * (1 + (mx_y - mn_y)/(mx_x - mn_x))) + fac_y * ((b * (1 + (mx_x - mn_x)/(mx_y - mn_y))))))
        return cost

    # default translation cost function where
        # a is the translation along to the x axis
        # b is the translation along to the y axis
    @staticmethod
    def _translation_cost(a, b, ln, fac_x=0.0, fac_y=0.0):
        a = abs(a)
        b = abs(b)
        return ln * (fac_x * a + fac_y * b)

    # default translation cost function where
        # a is the translation along to the x axis
        # b is the translation along to the y axis
    @staticmethod
    def _scaling_cost(a, b, ln, fac_x=5, fac_y=5, flip_x=-1, flip_y=-1):
        if flip_x == -1:
            flip_x = fac_x * 5
        if flip_y == -1:
            flip_y = fac_y * 5
        if a < 0:
            fac_x = flip_x
        if b < 0:
            fac_y = flip_y

        a = abs(a)
        b = abs(b)
        if a < 1:
            a = 1/a
        if b < 1:
            b = 1/b

        return ln * (fac_x * a + fac_y * b)



    # default rotation cost functionreg.total_cost(reg.original_obj[], t)
    @staticmethod
    def _rotation_cost(r, ln, fac_r=10):
        r = abs(r)
        cost = ln * (fac_r * r)
        return cost

    # for given parameters obtain the transformation matrix, according to the following order
    #   p[0]: the scaling the x direction
    #   p[1]: the scaling the y direction
    #   p[2]: rotation for theta degrees (counter clock-wise in radian)
    #   p[3]: shearing in the x axis
    #   p[4]: shearing in the y axis
    #   p[5]: translation in the x direction
    #   p[6]: translation in the y direction

    @staticmethod
    def obtain_transformation_matrix(p):
        t = np.zeros(6)
        t[0] = p[0] * (np.cos(p[2]) * (1 + p[3] * p[4]) - p[4] * np.sin(p[2]))
        t[1] = p[0] * (p[3] * np.cos(p[2]) - np.sin(p[2]))
        t[2] = p[5]
        t[3] = p[1] * (np.sin(p[2]) * (1 + p[3] * p[4]) + p[4] * np.cos(p[2]))
        t[4] = p[1] * (p[3] * np.sin(p[2]) + np.cos(p[2]))
        t[5] = p[6]
        return t

    # calculate the turning angle based on three points coordinates
    @staticmethod
    def calc_turning(x0, y0, x1, y1, x2, y2) -> float:
        dot = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)
        det = (x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1)
        if dot == 0 or det == 0:
            return 0.0
        return np.arctan2(-det, -dot) * 180.0 / np.pi + 180

    # change the coordinates of obj1, obj2 by translating them by xo, and yo
    @staticmethod
    def change_cords(obj1, obj2, xo, yo):
        obj1.transform([1, 0, xo, 0, 1, yo])
        obj2.transform([1, 0, xo, 0, 1, yo])

    # change the coordinates of both objects according the origin_x, origin_y, of ref_obj
    # t represent the direction of the change
    @staticmethod
    def normalize_coords(ref_obj, tar_obj, t):
        xo, yo = ref_obj.origin_x, ref_obj.origin_y
        RegistrationUtils.change_cords(ref_obj, tar_obj, t * xo, t * yo)

    # obtain sequential translation matrix according to translation parameters of shearing, rotation, scaling, and translation
    @staticmethod
    def get_seq_translation_matrices(p):
        tmp = []
        # shearing parallel to y
        tmp.append([1.0, 0.0, 0.0, p[4], 1.0, 0.0])
        # shearing parallel to x
        tmp.append([1.0, p[3], 0.0, 0.0, 1.0, 0.0])
        # rotation
        tmp.append([np.cos(p[2]), -np.sin(p[2]), 0.0, np.sin(p[2]), np.cos(p[2]), 0.0])
        # scaling
        tmp.append([p[0], 0.0, 0.0, 0.0, p[1], 0.0])
        # translation
        tmp.append([1.0, 0.0, p[5], 0.0, 1.0, p[6]])

        return tmp

    # transfer list of x, y according to t
    @staticmethod
    def transfer(x, y, t):
        for i in range(len(x)):
            xx, yy = x[i], y[i]
            x[i] = t[0] * x[i] + t[1] * y[i] + t[2]
            y[i] = t[3] * xx + t[4] * yy + t[5]
        return x, y

    # calculate the dissimilarity for the obj1, obj2 after transforming the obj1 accrding the parameter p (transformation cost not included)
    # p has 7 parameters:
        # p[0]: the scaling the x direction
        # p[1]: the scaling the y direction
        # p[2]: rotation for theta degrees (counter clock-wise in radian)
        # p[3]: shearing in the x axis
        # p[4]: shearing in the y axis
        # p[5]: translation in the x direction
        # p[6]: translation in the y direction
    @staticmethod
    def calc_dissimilarity(ref_obj:UnlabeledObject, tar_obj:UnlabeledObject, p, target_nn = None):
        x, y = np.array(copy.deepcopy(ref_obj.get_x())), np.array(copy.deepcopy(ref_obj.get_y()))
        x1, y1 = np.array(copy.deepcopy(tar_obj.get_x())), np.array(copy.deepcopy(tar_obj.get_y()))

        # transform the org_obj
        t = RegistrationUtils.obtain_transformation_matrix(p)
        x, y = RegistrationUtils.transfer(x, y, t)

        x = list(map(lambda q: q if isinstance(q, np.float64) else q._value, x))
        y = list(map(lambda q: q if isinstance(q, np.float64) else q._value, y))
        
        # obtain KDtree of the target obj
        if target_nn == None:
            target_nn = Nearest_search(x1, y1)
        
        # obtain KDtree of the refrenced obj
        reference_nn = Nearest_search(x, y)
        
        tot = 0.0
        # find nearest point from the target object to the points of the referenced object
        tot += target_nn.query(x, y)

        # find nearest point from the referenced object to the ith point of the target object
        tot += reference_nn.query(x1, y1)
        return tot / (len(ref_obj) + len(tar_obj))

        
        # TODO: the following block is dedicated to take into account
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