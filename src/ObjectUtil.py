from xml.dom import minidom
from Point import Point
from LabeledObject import LabeledObject
from math import sqrt
from math import ceil
import math
import numpy as np
from Vector import Vector

class ObjectUtil():

    @staticmethod
    def xml_to_LabledObjects(file):

        data = minidom.parse(file)
        points = data.getElementsByTagName('point')
        strokes = data.getElementsByTagName('stroke')

        # create dictionary for id -> point
        point_dic = {}
        for el in points:
            x = int(el.attributes['x'].value)
            y = int(el.attributes['y'].value)
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

    @staticmethod
    def match_objects_size(obj1, obj2):

        if obj1.len() < obj2.len():
            obj1, obj2 = obj2, obj1

        if obj1.len() != obj2.len():
            d = obj1.len() - obj2.len()
            t = ceil(d/(obj2.len()-1))
            step = max(1, int((obj2.len()-1)/d))
            print(obj1.len(), obj2.len(), d, t, step)
            tem = []
            for i in range(0, obj2.len()-1):
                tem.append(obj2.lst[i])
                if i % step == 0 and d > 0:
                    # add dummy points
                    x1, y1 = obj2.lst[i].x, obj2.lst[i].y
                    x2, y2 = obj2.lst[i+1].x, obj2.lst[i+1].y
                    tm = obj2.lst[i].t
                    kx, ky = (x2-x1)/(t+1), (y2-y1)/(t+1)
                    eps = float(0.0001)
                    x, y, tm = x1 + kx, y1 + ky, tm+eps
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


    @staticmethod
    def __calc_slop_RMSE(ind1, ind2, x1, y1, x2, y2):
        ln = len(x1)
        tot = tot2 = 0
        for i in range(0, ln):
            slop1 = slop2 = 0
            if (x1[(ind1+i)%ln]-x1[ind1]) != 0:
                slop1 = (y1[(ind1+i)%ln]-y1[ind1])/(x1[(ind1+i)%ln]-x1[ind1])
            if (x2[(ind2+i)%ln]-x2[ind2]) != 0:
                slop2 = (y2[(ind2+i)%ln]-y2[ind2])/(x2[(ind2+i)%ln]-x2[ind2])

            tot += sqrt((y2[(ind2+i)%ln] - y1[(ind1+i)%ln] - y2[ind2] + y1[ind1])**2)
            tot += sqrt((x2[(ind2+i)%ln] - x1[(ind1+i)%ln] - x2[ind2] + x1[ind1])**2)
            tot2 += sqrt((y2[(ind2 + i) % ln] - y1[(ind1 - i + ln) % ln] - y2[ind2] + y1[ind1]) ** 2)
            tot2 += sqrt((x2[(ind2 + i) % ln] - x1[(ind1 - i + ln) % ln] - x2[ind2] + x1[ind1]) ** 2)
            #tot += sqrt((slop1 - slop2) ** 2)

        return min(tot, tot2)

    @staticmethod
    def match_degree(obj1, obj2):
        # math the number of points of the two objects
        ObjectUtil.match_objects_size(obj1, obj2)
        x1, y1, x2, y2 = obj1.get_x(), obj1.get_y(), obj2.get_x(), obj2.get_y()

        # calculate the minimum RMSE
        mn_rms = 10000000000
        a1 = b1 = a2 = b2 = 0
        for i in range(len(x2)):
            for j in range(len(x2)):
                tmp = ObjectUtil.__calc_slop_RMSE(i, j, x1, y1, x2, y2)
                if tmp < mn_rms:
                    mn_rms = tmp
                    a1, b1, a2, b2 = x1[i], y1[i], x2[j], y2[j]


        return mn_rms, a1, b1, a2, b2


    @staticmethod
    def calc_turning(a : Point, b : Point, c : Point) -> float:
        ba = Vector(b, a)
        bc = Vector(b, c)
        return math.acos(ba * bc / (len(ba) + len(bc)))

