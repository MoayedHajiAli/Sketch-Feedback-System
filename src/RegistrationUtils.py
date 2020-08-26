from Stroke import Stroke
from Nearest_search import Nearest_search
from Registration import RegisterTwoObjects
import numpy as np
from UnlabeledObject import UnlabeledObject
import copy
from Nearest_search import Nearest_search
from ObjectUtil import ObjectUtil

class RegistrationUtils:
    """this static class provides basic functuionality for registering
    the objects  

    Methods
    -------
    obtain_transformation_matrix (array-type of shape (7)) -> array-type of shape(6)
        receives tansformation parameters of the form (scaling-x, scaling-y, roation, 
        shearing-x, shearing-y, translation-x, translation-y) and returns the 
        transformation parameters

    calc_turning(x0, y0, x1, y1, x2, y2) -> float
        returns the turning angles constructed in the given three points in degree and
        in the range (0, 360)

    change_cords(obj1:UnlabeledObject, obj2:UnlabeledObject, xo, yo)
        translate both given objects xo in x-direction and yo in y-direction
    
    normalize_coords(ref_obj:UnlabeledObject, tar_obj:UnlabeledObject, t)
        transform both objects to the origin of the ref_obj, the left-most point of an
        object is considered as the origin of that object. it is stored in the objcet as
        UnlabeledObject.origin_x, and UnlabeledObject.origin_y.
    
    get_seq_translation_matrices(array-type of shape(7)) -> array-type of shape(5,6)
        receives tansformation parameters of the form (scaling-x, scaling-y, roation, 
        shearing-x, shearing-y, translation-x, translation-y) and breakthrough these 
        parameters into 5 transformation metrices of  shearing, rotation, scaling, and
        translation

    transfer(x:array-type of shape(n), y:array-type of shape(n), t:array-type of shape(6))
        transform the coordinates x, and y according the transformation params t
    
    calc_dissimilarity(ref_obj:UnlabeledObject, tar_obj:UnlabeledObject, p:array-type of shape(7),
                      target_nn:Nearest_search = None, turning_ang=False, cum_and=False, distance=False)
        calculate the dissimilarity of two objects after transforming ref_obj according to 
        parameters p which are of the order (scaling-x, scaling-y, roation, shearing-x, 
        shearing-y, translation-x, translation-y).
        Dissimilarity is calculated based on the distance of the nearest point of the opposite 
        object for every given point the object      
    """
    
    inf = 1e9+7
    # default shearing cost function where
        # a is the shearing parallel to the x axis
        # b is the shearing parallel to the y axis
    @staticmethod
    def _shearing_cost(a, b, mn_x, mn_y, mx_x, mx_y, ln, fac_x=15, fac_y=15):
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
    def _rotation_cost(r, ln, fac_r=12):
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
        return np.arctan2(-det, -dot) * 180.0 / np.pi

    # change the coordinates of obj1, obj2 by translating them by xo, and yo
    @staticmethod
    def change_cords(obj1:UnlabeledObject, obj2:UnlabeledObject, xo, yo):
        obj1.transform([1, 0, xo, 0, 1, yo])
        obj2.transform([1, 0, xo, 0, 1, yo])

    # change the coordinates of both objects according the origin_x, origin_y, of ref_obj
    # t represent the direction of the change
    @staticmethod
    def normalize_coords(ref_obj:UnlabeledObject, tar_obj:UnlabeledObject, t):
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
            x[i] = t[0] * xx + t[1] * yy + t[2]
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
    def calc_dissimilarity(ref_obj:UnlabeledObject, tar_obj:UnlabeledObject, p, target_nn:Nearest_search = None, turning_ang=False,
     cum_ang=False, length=False, turning_fac = 0.05, cum_fac = 0.3, len_fac = 0.01):
        x, y = np.array(copy.deepcopy(ref_obj.get_x())), np.array(copy.deepcopy(ref_obj.get_y()))
        x1, y1 = np.array(copy.deepcopy(tar_obj.get_x())), np.array(copy.deepcopy(tar_obj.get_y()))

        # transform the org_obj
        t = RegistrationUtils.obtain_transformation_matrix(p)
        x, y = RegistrationUtils.transfer(x, y, t)

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

        if not turning_ang and not cum_ang and not length:
            return tot / (len(ref_obj) + len(tar_obj))
 
        # cum1 = RegistrationUtils.calc_turning(x[0] - 1, y[0], x[0], y[0], x[1], y[1])
        # cum2 = RegistrationUtils.calc_turning(x1[0] - 1, y1[0], x1[0], y1[0], x1[1], y1[1])
        # ang = min(abs(cum2 - cum1), 360.0 - (cum2 - cum1))
        
        # if turning_ang:
        #     tot += turning_fac * ang

        # if cum_ang:
        #     tot += cum_fac * ang
        cum1 = cum2 = 0

        # i, j represent the current points of the target and the referenced objects respectively
        j, i = 0, target_nn.query_ind(x[0], y[0])
        for _ in range(len(ref_obj)):
            if j + 2 < len(x1):
                i_1 = (i - 1 + len(tar_obj)) % len(tar_obj)
                i_2 = (i - 2 + len(tar_obj)) % len(tar_obj)
                t1 = RegistrationUtils.calc_turning(x[j], y[j], x[j + 1], y[j + 1], x[j + 2], y[j + 2])
                t2 = RegistrationUtils.calc_turning(x1[i], y1[i], x1[i_1], y1[i_1], x1[i_2], y1[i_2])
                if turning_ang:
                    print(t1, t2)
                    ang = abs(t1 - t2)
                    tot += turning_fac * ang * (len(ref_obj) + len(tar_obj))
        

                if cum_ang:
                    cum1 = (t1 + cum1) % 180
                    cum2 = (t2 + cum2) % 180
                    print(cum1, cum2)
                    ang = abs(cum2 - cum1)
                    tot += cum_fac * ang
        
            if length and j + 1 < len(ref_obj):
                i_1 = (i - 1 + len(tar_obj)) % len(tar_obj)
                ln1 = np.sqrt((x[j + 1] - x[j]) ** 2 + (y[j + 1] - y[j]) ** 2)
                ln2 = np.sqrt((x1[i_1] - x1[j]) ** 2 + (y1[i_1] - y1[i]) ** 2)
                tot += len_fac * abs(ln2 - ln1)
            j += 1
            i = (i + 1 + len(tar_obj)) % len(tar_obj)
        return tot / (len(ref_obj) + len(tar_obj))
        
    @staticmethod
    def identify_similarity(obj1:UnlabeledObject, obj2:UnlabeledObject, t) -> bool:
        obj1 = copy.deepcopy(obj1)
        obj2 = copy.deepcopy(obj2)
        d, t = RegisterTwoObjects(obj1, obj2, lambda p : 0).optimize(t)
        obj1.transform(t)
        strokes = []
        obj1_kd = Nearest_search(obj1.get_x, obj2.get_y)
        for s in obj2.get_strokes():
            tmp = []
            for p in s.get_points():
                tmp.append(obj1_kd.query_point(p))
            
            strokes.append(Stroke(tmp))