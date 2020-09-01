from xml.dom import minidom
from Point import Point
from Stroke import Stroke
from math import sqrt
from math import ceil
import numpy as np
from scipy.interpolate import interp1d
from UnlabeledObject import UnlabeledObject
import copy
import warnings

class ObjectUtil:

    # for a given file, read the file and transform it to an array of strokes
    @staticmethod
    def xml_to_Strokes(file, mn_len=0, flip=False, shift_x=0.0, shift_y=0.0):

        data = minidom.parse(file)
        points = data.getElementsByTagName('point')
        strokes = data.getElementsByTagName('stroke')
        objects = data.getElementsByTagName('Object')

        # create dictionary for id -> point
        point_dic = {}
        for el in points:
            x = float(el.attributes['x'].value)
            y = float(el.attributes['y'].value)
            time = float(el.attributes['time'].value)

            if flip:
                y *= -1

            x += shift_x
            y += shift_y

            point_dic[el.attributes['id'].value] = (x, y, time)

        all_strokes, strokes_id, strokes_collections, labels = [], [], [], []
        # extract strokes
        for st in strokes:
            pts = st.getElementsByTagName('arg')
            strokes_id.append(st.attributes['id'].value)
            stroke = []
            for pt in pts:
                id = pt.firstChild.nodeValue
                x, y, t = point_dic[id]
                stroke.append(Point(x, y, t))
            if len(stroke) >= mn_len:
                all_strokes.append(Stroke(stroke))
        for o in objects:
            labels.append(o.attributes['Label'].value)
            sts = o.getElementsByTagName('arg')
            tmp = []
            for s in sts:
                id = s.firstChild.nodeValue
                tmp.append(strokes_id.index(id))
            strokes_collections.append(tmp)

        return np.array(all_strokes), strokes_collections, labels

    # for a given file, read the file and transform it to an array of objects
    @ staticmethod
    def xml_to_UnlabeledObjects(file, mn_len=0, re_sampling=1.0, flip=False, shift_x=0.0, shift_y = 0.0):
        strokes, strokes_collections, labels = ObjectUtil.xml_to_Strokes(file, mn_len=mn_len, flip=flip, shift_x=shift_x, shift_y=shift_y)
        objs = ObjectUtil.collect_strokes(strokes, strokes_collections)

        # re-sample the objects
        if re_sampling != 0.0:
            for i in range(len(objs)):
                objs[i] = ObjectUtil.object_restructure(objs[i], re_sampling)
        return objs, labels

    # for given two objects, match their size by adding dummy points
    @staticmethod
    def match_objects_size(obj1, obj2):
        if obj1.len() < obj2.len():
            obj1, obj2 = obj2, obj1

        if obj1.len() != obj2.len():
            d = obj1.len() - obj2.len()
            t = ceil(d / (obj2.len() - 1))
            step = max(1, int((obj2.len() - 1) / d))
            tem = []
            for i in range(0, obj2.len() - 1):
                tem.append(obj2.lst[i])
                if i % step == 0 and d > 0:
                    # add dummy points
                    x1, y1 = obj2.lst[i].x, obj2.lst[i].y
                    x2, y2 = obj2.lst[i + 1].x, obj2.lst[i + 1].y
                    tm = obj2.lst[i].t
                    kx, ky = (x2 - x1) / (t + 1), (y2 - y1) / (t + 1)
                    eps = float(0.0001)
                    x, y, tm = x1 + kx, y1 + ky, tm + eps
                    for _ in range(t):
                        if d <= 0:
                            break
                        tem.append(Point(x, y, tm))
                        d -= 1
                        x += kx
                        y += ky
                        tm += eps
            tem.append(obj2.lst[-1])
            obj2.lst = tem

    # calculating dissimilarity based on ordered points coordinates RMSE // Deprecated
    @staticmethod
    def __calc_dissimilarity(ind1, ind2, x1, y1, x2, y2):
        ln = len(x1)
        tot = tot2 = 0
        for i in range(0, ln):
            slop1 = slop2 = 0
            if (x1[(ind1 + i) % ln] - x1[ind1]) != 0:
                slop1 = (y1[(ind1 + i) % ln] - y1[ind1]) / (x1[(ind1 + i) % ln] - x1[ind1])
            if (x2[(ind2 + i) % ln] - x2[ind2]) != 0:
                slop2 = (y2[(ind2 + i) % ln] - y2[ind2]) / (x2[(ind2 + i) % ln] - x2[ind2])

            tot += sqrt((y2[(ind2 + i) % ln] - y1[(ind1 + i) % ln] - y2[ind2] + y1[ind1]) ** 2)
            tot += sqrt((x2[(ind2 + i) % ln] - x1[(ind1 + i) % ln] - x2[ind2] + x1[ind1]) ** 2)
            tot2 += sqrt((y2[(ind2 + i) % ln] - y1[(ind1 - i + ln) % ln] - y2[ind2] + y1[ind1]) ** 2)
            tot2 += sqrt((x2[(ind2 + i) % ln] - x1[(ind1 - i + ln) % ln] - x2[ind2] + x1[ind1]) ** 2)
            # tot += sqrt((slop1 - slop2) ** 2)

        return min(tot, tot2)

    # find the optimal translation transformation by iterating over all pair of points
    @staticmethod
    def find_optimal_translation(obj1, obj2):
        # math the number of points of the two objects
        ObjectUtil.match_objects_size(obj1, obj2)
        x1, y1, x2, y2 = obj1.get_x(), obj1.get_y(), obj2.get_x(), obj2.get_y()

        # calculate the minimum RMSE
        mn_rms = 10000000000
        a1 = b1 = a2 = b2 = 0
        for i in range(len(x2)):
            for j in range(len(x2)):
                tmp = ObjectUtil.__calc_dissimilarity(i, j, x1, y1, x2, y2)
                if tmp < mn_rms:
                    mn_rms = tmp
                    a1, b1, a2, b2 = x1[i], y1[i], x2[j], y2[j]

        return mn_rms, a1, b1, a2, b2

    # for a given object, calculate its perimeter
    @staticmethod
    def calc_perimeter(obj) -> float:
        tot, i = 0, 1
        while i < len(obj):
            tot += Point.euclidean_distance(obj.get_points()[i], obj.get_points()[i-1])
            i += 1

        return tot

    # for a given stroke, and target number of points, restructure the objects by placing
    # the points at equal distances
    @staticmethod
    def stroke_restructure(stroke:Stroke, n) -> Stroke:
        perimeter = ObjectUtil.calc_perimeter(stroke)
        p = perimeter / (n)
        # print("Before", len(stroke), perimeter, p)

        # given two points, return a placed new point at a distance x between p1, p2
        def _place(p1, p2, x):
            f = interp1d([p1.x, p2.x], [p1.y, p2.y])
            d = p2.x - p1.x
            td = p2.t - p1.t
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    r = x / Point.euclidean_distance(p1, p2)
                except Warning:
                    print(x, Point.euclidean_distance(p1, p2))
            return Point(p1.x + d * r, f(p1.x + d * r), p1.t + td * r)

        # list to hold the new points
        lst = [copy.deepcopy(stroke.get_points()[0])]

        # j: remaining distance to be traversed from the last iteration
        j = p
        tot = 0
        for i in range(len(stroke) - 1):
            p1, p2 = stroke.get_points()[i], stroke.get_points()[i + 1]
            d = Point.euclidean_distance(p1, p2)
            if d == 0:
                continue
            # c: how much distance from the first point has been traversed toward the second point
            c = 0
            while c + j <= d:
                c += j
                tot += j
                j = p
                lst.append(_place(p1, p2, c))
            tot += d - c
            j -= d - c

        new_stroke = Stroke(lst)
        perimeter = ObjectUtil.calc_perimeter(new_stroke)
        # print("After", len(new_stroke), perimeter, tot)
        return new_stroke

    @staticmethod
    def object_restructure(obj:UnlabeledObject, ratio=1.0, n=0, mn_len=10) -> UnlabeledObject:
        if n == 0:
            n = len(obj) * ratio
        peri = ObjectUtil.calc_perimeter(obj)
        tmp_lst = []
        for stroke in obj.get_strokes():
            m = max(ObjectUtil.calc_perimeter(stroke) / peri * n, mn_len)
            tmp_lst.append(ObjectUtil.stroke_restructure(stroke, m))
        return UnlabeledObject(tmp_lst)


    # obtain object from strokes according to collections matrix
    # collection matrix should have a shape of (n objects, k strokes for each object)
    # the number of strokes can differ from one object to another
    @staticmethod
    def collect_strokes(strokes, collections):
        strokes = np.array(strokes)
        object_lst = []
        for col in collections:
            object_lst.append(UnlabeledObject(strokes[col]))
        return object_lst
