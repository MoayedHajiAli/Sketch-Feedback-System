from xml.dom import minidom
from Point import Point
from LabeledObject import LabeledObject
from math import sqrt
from math import ceil
import math
import numpy as np
from Vector import Vector
from scipy.interpolate import interp1d



class ObjectUtil():

    # for a given file, read the file and transform it to an array of objects
    @staticmethod
    def xml_to_LabledObjects(file):

        data = minidom.parse(file)
        points = data.getElementsByTagName('point')
        strokes = data.getElementsByTagName('stroke')

        # create dictionary for id -> point
        point_dic = {}
        for el in points:
            x = float(el.attributes['x'].value)
            y = float(el.attributes['y'].value)
            time = float(el.attributes['time'].value)
            point_dic[el.attributes['id'].value] = (x, y, time)

        objects = []
        # extract strokes
        for st in strokes:
            pts = st.getElementsByTagName('arg')
            stroke = []
            for pt in pts:
                id = pt.firstChild.nodeValue
                x, y, t = point_dic[id]
                stroke.append(Point(x, y, t))
            objects.append(LabeledObject(stroke))
        return np.array(objects)

    # for given two objects, match their size by adding dummy points
    @staticmethod
    def match_objects_size(obj1, obj2):
        if obj1.len() < obj2.len():
            obj1, obj2 = obj2, obj1

        if obj1.len() != obj2.len():
            d = obj1.len() - obj2.len()
            t = ceil(d / (obj2.len() - 1))
            step = max(1, int((obj2.len() - 1) / d))
            print(obj1.len(), obj2.len(), d, t, step)
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

    # calculating dissimilarity based on ordered points coordinates RMSE
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
    def calc_perimeter(obj:LabeledObject) -> float:
        tot, i = 0, 1
        while i < len(obj):
            tot += Point.euclidean_distance(obj.get_points()[i], obj.get_points()[i-1])
            i += 1

        return tot

    # for a given object, and target number of points, restructure the objects by placing
    # the points are equal distance
    @staticmethod
    def stroke_restructure(obj:LabeledObject, n:int) -> LabeledObject:
        perimeter = ObjectUtil.calc_perimeter(obj)
        p = perimeter / (n - 1)

        # given two points, return a placed new point at a distance x between p1, p2
        def _place(p1, p2, x):
            f = interp1d([p1.x, p2.x], [p1.y, p2.y])
            d = p2.x - p1.x
            td = p2.t - p1.t
            r = x / Point.euclidean_distance(p1, p2)
            return Point(p1.x + d * r, f(p1.x + d * r), p1.t + td * r)

        # list to hold the new points
        lst = [obj.get_points()[0]]

        # j: remaining distance to be traversed from the last iteration
        j = p
        for i in range(len(obj)-1):
            p1, p2 = obj.get_points()[i], obj.get_points()[i+1]
            d = Point.euclidean_distance(p1, p2)
            # c: how much distance from the first point has been traversed toward the second point
            c = 0
            while c + j <= d:
                c += j
                j = p
                lst.append(_place(p1, p2, c))
            j -= d - c

        return LabeledObject(lst)



