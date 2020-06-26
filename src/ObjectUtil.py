from xml.dom import minidom
from Point import Point
from LabeledObject import LabeledObject
from math import sqrt
from math import ceil
import numpy as np

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
    def match_size(obj1, obj2):

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
                    tm2 = obj2.lst[i+1].t
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
        print("tst", obj1.len(), obj2.len())


    @staticmethod
    def match_degree(obj1, obj2):

        # math the number of points of the two objects
        ObjectUtil.match_size(obj1, obj2)
        x1, y1, x2, y2 = obj1.get_x(), obj1.get_y(), obj2.get_x(), obj2.get_y()
        x1 = x1 + x1
        y1 = y1 + y1

        print("Match", len(x1), len(x2))

        # calculate the minimum RMSE
        mn_RMS, ind = 1000, 0
        for i in range(len(x2)):
            k, RMS = i+1, 0
            for j in range(1, len(x2)):
                slop1, slop2 = 0, 0
                if (x1[k] - x1[k - 1]) != 0:
                    slop1 = (y1[k] - y1[k - 1]) / (x1[k] - x1[k - 1])
                if (x2[j] - x2[j - 1]) != 0:
                    slop2 = (y2[j] - y2[j - 1]) / (x2[j] - x2[j - 1])
                if slop1 != 0:
                    slop1/=slop1
                if slop2 != 0:
                    slop2/=slop2
                RMS += sqrt((slop1-slop2)**2)
                k += 1
            RMS /= len(x2)
            if RMS < mn_RMS:
                mn_RMS, ind = RMS, i

        return mn_RMS, x1[ind], y1[ind], x2[0], y2[0]