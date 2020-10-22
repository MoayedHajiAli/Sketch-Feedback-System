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
from sketchformer.basic_usage.sketchformer import continuous_embeddings
from rdp import rdp


class ObjectUtil:

    """for a given file, read the file and transform it to an dictonary of points
    Params:
        flip : flip the image vertically
        mn_len : ignore any strokes with length smaller than mn_len
        shift_x : inital horizental shift
        shift_y : inital vertical shift

    Returns:
        Dictionary: id -> points dictionary
    """

    sketchformer = None


    @staticmethod
    def xml_to_PointsDict(file, flip = False, shift_x=0.0, shift_y=0.0):
        data = minidom.parse(file)
        points = data.getElementsByTagName('point')

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
        
        return point_dic
    
    @staticmethod
    def xml_to_points(file, flip = False, shift_x=0.0, shift_y=0.0):
        return list(ObjectUtil.xml_to_PointsDict(file=file, flip=flip, shift_x=shift_x, shift_y=shift_y).values())

    """for a given file, read the file and transform it to a dictionary of strokes
    Params:
        flip : flip the image vertically
        mn_len : ignore any strokes with length smaller than mn_len
        shift_x : inital horizental shift
        shift_y : inital vertical shift

    Returns:
        Dictionary: id -> strokes dictionary
    """
    @staticmethod
    def xml_to_StrokesDict(file, re_sampling=1.0, mn_len=0, flip=False, shift_x=0.0, shift_y=0.0):
        data = minidom.parse(file)
        strokes = data.getElementsByTagName('stroke')

        # create dictionary for id -> point
        point_dic = ObjectUtil.xml_to_PointsDict(file, flip=flip, shift_x=shift_x, shift_y=shift_y)

        # create dictionary for id -> stroke
        stroke_dict = {}
        for st in strokes:
            pts = st.getElementsByTagName('arg')
            strokes_id = st.attributes['id'].value
            pt_lst = []
            for pt in pts:
                id = pt.firstChild.nodeValue
                x, y, t = point_dic[id]
                pt_lst.append(Point(x, y, t))
            if len(pt_lst) >= mn_len:
                stroke_dict[strokes_id] = Stroke(pt_lst)
        
        # re-sample the objecte
        # TODO: re_sampling will contradict with the rdp. Choose one. Now we will assume that the resampling as always 0.0
        if re_sampling != 0.0:
            for id in stroke_dict:
                stroke_dict[id] = ObjectUtil.stroke_restructure(stroke_dict[id], max(mn_len, int(re_sampling * len(stroke_dict[id]))))

        return stroke_dict

    @staticmethod
    def xml_to_strokes(file, re_sampling = 1.0, mn_len=0, flip=False, shift_x=0.0, shift_y=0.0):
        return list(ObjectUtil.xml_to_StrokesDict(file, re_sampling = re_sampling, mn_len=mn_len, flip=flip, shift_x=shift_x, shift_y=shift_y).values())
    
    """for a given file, find all objects and their labels

    Params:
    flip : flip the image vertically
    mn_len : ignore any strokes with length smaller than mn_len
    shift_x : inital horizental shift
    shift_y : inital vertical shift
    re_sampling: re-draw the object with fewer/more number of points with a uniform distribuion of points
                along the perimeter. if re_sampling equals 0.0, then no re_drawing will happen.

    Returns:
        Array-like(N): array of objects
        Array-line(N): labels to the returned objects 
    """
    @ staticmethod
    def xml_to_UnlabeledObjects(file, mn_len=0, re_sampling=1.0, flip=False, shift_x=0.0, shift_y=0.0, rdp=True):
        data = minidom.parse(file)
        objects = data.getElementsByTagName('Object')

        # obtian stroke dict
        strokes_dict = ObjectUtil.xml_to_StrokesDict(file, re_sampling=re_sampling, mn_len=mn_len, flip=flip, shift_x=shift_x, shift_y=shift_y)

        objs, labels = [], []
        for o in objects:
            labels.append(o.attributes['Label'].value)
            sts = o.getElementsByTagName('arg')
            strokes_lst = []
            for s in sts:
                id = s.firstChild.nodeValue
                strokes_lst.append(strokes_dict[id])

            objs.append(UnlabeledObject(strokes_lst))
        
        if rdp:
            objs = ObjectUtil.reduce_rdp(objs)

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
    
    @staticmethod
    def convert_to_stroke3(sketches):
        """convert the given sketches to stroke-3 (x, y, p) format, where x, y are 
         the point relative position to the previous point together with its binary pen state.
         for any two consecutive strokes, a point in the middle will be added with a pen state 1 (pen lifted)
        Args:
            sketches (list): list of sketches to be converted
        
        Returns: the converted sketches to stroke-3 format and store them in arrays
        """
        converted_sketches = []
        for sketch_con in sketches:
            sketch = copy.deepcopy(sketch_con)
            strokes_lst = sketch.get_strokes()

            # gather all the points in one stroke
            tmp_stroke_3 = []
            for i in range(len(strokes_lst)):
                if i != 0:
                    # insert an imaginary point between the end point of stroke i-1 and the first point in
                    # stroke i, with a pen lifted state
                    p1 = strokes_lst[i - 1].get_points()[-1]
                    p2 = strokes_lst[i].get_points()[0]
                    tmp_stroke_3.append([(p1.x + p2.x) / 2, (p1.y + p2.y) / 2, 1])
                
                # add all stroke's points with state 0
                for p in strokes_lst[i].get_points():
                    tmp_stroke_3.append([p.x, p.y, 0])
            
            converted_sketches.append(tmp_stroke_3)

        return converted_sketches


    @staticmethod
    def get_embedding(objs):
        """extract the sketchformer embedding for the given sketch in (x, y, time) format 

        Args:
            sketches ([list]): list of sketches in the conitnuious format of (x, y, time)
        """
        # get the pre-trained model
        if ObjectUtil.sketchformer is None:
            ObjectUtil.sketchformer = continuous_embeddings.get_pretrained_model()

        # obtain the stroke-3 format of the sketches
        objs = ObjectUtil.convert_to_stroke3(objs)

        # return the embeddings
        return ObjectUtil.sketchformer.get_embeddings(objs)

    @staticmethod
    def classify(objs):
        """classify the given sketches of the (x, y, time) format 

        Args:
            sketches ([list]): list of sketches in the conitnuious format of (x, y, time)
        """
        # get the pre-trained model
        if ObjectUtil.sketchformer is None:
            ObjectUtil.sketchformer = continuous_embeddings.get_pretrained_model()

        # obtain the stroke-3 format of the sketches
        objs = ObjectUtil.convert_to_stroke3(objs)

        # return the embeddings
        return ObjectUtil.sketchformer.classify(objs)

    @staticmethod
    def reduce_rdp(objs):
        """reduce the given set of UnlabeledObjects using Ramer–Douglas–Peucker algorithm

        Args:
            objs ([list]): list of objects to be reduced

        Returns:
            [list]: list of the reduced objects 
        """
        reduced_objs = []

        for obj in objs:
            reduced_strokes = []
            for stroke in obj.get_strokes():
                tmp = [[pt.x, pt.y] for pt in stroke.get_points()]
                
                # reduce the strokes points
                reduced_points = rdp(tmp)
                
                # obtain the matches points to the reduced points 
                ind, res_points = 0, []
                for pt in stroke.get_points():
                    if ind >= len(reduced_points):
                        break

                    if pt.x == reduced_points[ind][0] and pt.y == reduced_points[ind][1]:
                        res_points.append(pt)
                        ind += 1
            
                reduced_strokes.append(Stroke(res_points))
            
            reduced_objs.append(UnlabeledObject(reduced_strokes))
        
        return reduced_objs
         

                



    

