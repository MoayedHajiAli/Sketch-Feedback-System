from xml.dom import minidom
from sketch_object.Point import Point
from sketch_object.Stroke import Stroke
from math import sqrt
from math import ceil
import numpy as np
from scipy.interpolate import interp1d
from sketch_object.UnlabeledObject import UnlabeledObject
import copy
import warnings
from sketchformer.basic_usage.sketchformer import continuous_embeddings
from rdp import rdp
import pathlib

class ObjectUtil:
    sketchformer = None
    @staticmethod
    def xml_to_pointsDict(file, flip = False, shift_x=0.0, shift_y=0.0):
        """for a given file, read the file and transform it to an dictonary of points where for each point id, return x, y and time
        Params:
            flip : flip the image vertically
            mn_len : ignore any strokes with length smaller than mn_len
            shift_x : inital horizental shift
            shift_y : inital vertical shift

        Returns:
            Dictionary: id -> Points dictionary
        """

        data = minidom.parse(file)
        points = data.getElementsByTagName('point')

        # create dictionary for id -> Point
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
        return list(ObjectUtil.xml_to_pointsDict(file=file, flip=flip, shift_x=shift_x, shift_y=shift_y).values())


    @staticmethod
    def xml_to_StrokesDict(file, re_sampling=0.0, mn_len=5, flip=False, shift_x=0.0, shift_y=0.0):
        """for a given file, read the file and transform it to a dictionary of strokes where for each stroke id, we store a stoke 
        containing all of the points
        Params:
            flip : flip the image vertically
            mn_len : ignore any strokes with length smaller than mn_len
            shift_x : inital horizental shift
            shift_y : inital vertical shift

        Returns:
            Dictionary: id -> strokes dictionary
        """
        data = minidom.parse(file)
        strokes = data.getElementsByTagName('stroke')

        # create dictionary for id -> Point
        point_dic = ObjectUtil.xml_to_pointsDict(file, flip=flip, shift_x=shift_x, shift_y=shift_y)

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
        
        # # re-sample the objecte (we are currently resampling the whole object) TODO DELETE
        # if re_sampling != 0.0:
        #     for id in stroke_dict:
        #         if re_sampling > 1:
        #             # resample to a fixed length
        #             stroke_dict[id] = ObjectUtil.stroke_restructure(stroke_dict[id], re_sampling)
        #         elif re_sampling <= 1:
        #             stroke_dict[id] = ObjectUtil.stroke_restructure(stroke_dict[id], max(mn_len, int(re_sampling * len(stroke_dict[id]))))

        return stroke_dict

    @staticmethod
    def xml_to_strokes(file, re_sampling = 0.0, mn_len=5, flip=False, shift_x=0.0, shift_y=0.0):
        return list(ObjectUtil.xml_to_StrokesDict(file, re_sampling = re_sampling, mn_len=mn_len, flip=flip, shift_x=shift_x, shift_y=shift_y).values())
    
    
    @staticmethod
    def xml_to_UnlabeledObjects(file, mn_len=5, re_sampling=0.0, flip=False, shift_x=0.0, shift_y=0.0, rdp=False):
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
                if id in strokes_dict.keys():
                    strokes_lst.append(strokes_dict[id])

            obj = UnlabeledObject(strokes_lst)
            if re_sampling != 0.0:
                obj = ObjectUtil.object_restructure(obj, n=re_sampling)

            objs.append(obj)
        
        if rdp:
            objs = ObjectUtil.reduce_rdp(objs)

        return objs, labels
    
    @staticmethod
    def xml_to_IndexedUnlabeledObjects(file, mn_len=5, re_sampling=0.0, flip=False, shift_x=0.0, shift_y=0.0, rdp=False):
        """for a given file, find all objects and their labels. Include in the objects the strokes indices

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
        data = minidom.parse(file)
        objects = data.getElementsByTagName('Object')

        # obtian stroke dict
        strokes_dict = ObjectUtil.xml_to_StrokesDict(file, re_sampling=re_sampling, mn_len=mn_len, flip=flip, shift_x=shift_x, shift_y=shift_y)

        objs, labels, inds = [], [], []
        for o in objects:
            labels.append(o.attributes['Label'].value)
            sts = o.getElementsByTagName('arg')
            strokes_lst, inds_tmp = [], []
            for s in sts:
                id = s.firstChild.nodeValue
                if id in strokes_dict.keys():
                    strokes_lst.append(strokes_dict[id])
                    inds_tmp.append(list(strokes_dict.keys()).index(id))

            obj = UnlabeledObject(strokes_lst)
            if re_sampling != 0.0:
                obj = ObjectUtil.object_restructure(obj, n=re_sampling)
            
            objs.append(obj)
            inds.append(sorted(inds_tmp))
        
        if rdp:
            objs = ObjectUtil.reduce_rdp(objs)

        return objs, labels, inds


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
        tot = 0
        if isinstance(obj, UnlabeledObject):
            for stroke in obj.get_strokes():
                tot += ObjectUtil.calc_perimeter(stroke)

        else:
            i = 1
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
        # print("Before", len(stroke), perimeter, n, p)

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
        j, tot = p, 0
        i, en = 0, len(stroke)-1
        while i != en:
            p1, p2 = stroke.get_points()[i], stroke.get_points()[(i + 1) % len(stroke)]
            d = Point.euclidean_distance(p1, p2)
            if d == 0:
                i = (i + 1) % len(stroke)
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
            i = (i + 1) % len(stroke)

        new_stroke = Stroke(lst)
        perimeter = ObjectUtil.calc_perimeter(new_stroke)
        # print("After ", len(new_stroke), perimeter, tot)
        return new_stroke

    @staticmethod
    def object_restructure(obj:UnlabeledObject, n, mn_len=10) -> UnlabeledObject:
        if n < 1.0:
            n = len(obj) * n

        perimeter = ObjectUtil.calc_perimeter(obj)

        # determine the position of the starting stroke and start point as the one with min x TODO

        tmp_lst = []
        for stroke in obj.get_strokes():
            m = max((ObjectUtil.calc_perimeter(stroke) / perimeter) * n, mn_len)
            tmp_lst.append(ObjectUtil.stroke_restructure(stroke, m))
        return UnlabeledObject(tmp_lst)


    # obtain object from sketch_object.Strokes according to collections matrix
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
    def stroke3_to_poly(sketches):
        """convert array of sketches in stroke-3 format to array of UnlabeledObject
        time of drawing are set randomly with taking into account the order of the points 
        Args:
            sketches (list): list of sketches in stroke-3

        Returns:
            [list]: list of Unlabeled objects
        """
        ret = []
        for sketch in sketches:
            pt_lst, strokes = [], []
            last_tm = 0.0
            x, y = 0, 0
            # # append the omitted points 
            # pt_lst.append(Point(x, y, last_tm))
            # last_tm += 1

            for p in sketch:
                if p[2] == 1:
                    # obtain a new stroke
                    if len(pt_lst) > 0:
                        pt_lst.append(Point(p[0] + x, p[1] + y, last_tm))
                        last_tm += 1
                        x, y = p[0] + x, p[1] + y
                        strokes.append(Stroke(pt_lst))
                        pt_lst = []
                else:
                    pt_lst.append(Point(p[0] + x, p[1] + y, last_tm))
                    last_tm += 1
                    x, y = p[0] + x, p[1] + y
            if len(pt_lst) > 0:
                strokes.append(Stroke(pt_lst))
            
            ret.append(UnlabeledObject(strokes))
        return ret

    @staticmethod
    def get_normalization_parameters(obj):
        mx_w, mx_h, mn_w, mn_h = -1, -1, 1e9, 1e9
        for stroke in obj.get_strokes():
            # get dimentions
            mx_w = max(mx_w, max([p.x for p in stroke.get_points()]))
            mn_w = min(mn_w, min([p.x for p in stroke.get_points()]))
            mx_h = max(mx_h, max([p.y for p in stroke.get_points()]))
            mn_h = min(mn_h, min([p.y for p in stroke.get_points()]))

        w, h = mx_w - mn_w, mx_h - mn_h
        mx_wh = max(w, h)
        return mn_w, mn_h, mx_wh

    @staticmethod
    def denormalized_transformation(org_obj, tar_obj, t, scale=100.0):
        """Given a transformation matrix on the normalized objects, return the
        correponsing transformation matrix on the non-normalized objects.
        Attention: Scale should match with the normalization function

        Args:
            org_obj (UnlabeledObject): org objects
            tar_obj (UnlabeledObject): Tar objects
            t (array(6)): transformation parameters

        Returns:
            t array(6): resulting transformation matrix on the
        """

        # get the normalization parameters
        mn_x_0, mn_y_0, f0 = ObjectUtil.get_normalization_parameters(org_obj)
        mn_x_1, mn_y_1, f1 = ObjectUtil.get_normalization_parameters(tar_obj)
        
        t_res = np.zeros(6)
        t_res[0] = t[0] * f1 / f0
        t_res[1] = t[1] * f1 / f0
        t_res[2] = - t[0] * f1 * mn_x_0 / f0 - t[1] * f1 * mn_y_0 / f0 + f1 * t[2] / scale + mn_x_1 
        # t_res[2] = -t[0] * (mn_x_0 * f1 / f0 + f1 / 2) - t[1] * (mn_y_0 * f1 / f0 + f1 / 2) + t[2] * (f1 / (2 * scale)) + f1/2 + mn_x_1
        # t_res[2] = -f1/f0 * (t[0] * mn_x_0 + t[1] * mn_y_0) - f1 * (t[0] + t[1]) + 2*f1*t[2]/scale + mn_x_1 + 1

        t_res[3] = t[3] * f1 / f0
        t_res[4] = t[4] * f1 / f0
        t_res[5] = - t[3] * f1 * mn_x_0 / f0 - t[4] * f1 * mn_y_0 / f0 + f1 * t[5] / scale + mn_y_1 
        # t_res[5] = -t[3] * (mn_x_0 * f1 / f0 + f1 / 2) - t[4] * (mn_y_0 * f1 / f0 + f1 / 2) + t[5] * (f1 / (2 * scale)) + f1/2 + mn_y_1
        # t_res[5] = -f1/f0 * (t[3] * mn_x_0 + t[4] * mn_y_0) - f1 * (t[3] + t[4]) + 2*f1*t[5]/scale + mn_y_1 + 1

        return t_res
         
    def denormalized_transformation(org_obj, tar_obj, t, scale=100.0):
        """Given a transformation matrix on the normalized objects, return the
        correponsing transformation matrix on the non-normalized objects.
        Attention: Scale should match with the normalization function

        Args:
            org_obj (UnlabeledObject): org objects
            tar_obj (UnlabeledObject): Tar objects
            t (array(6)): transformation parameters

        Returns:
            t array(6): resulting transformation matrix on the
        """

        # get the normalization parameters
        mn_x_0, mn_y_0, f0 = ObjectUtil.get_normalization_parameters(org_obj)
        mn_x_1, mn_y_1, f1 = ObjectUtil.get_normalization_parameters(tar_obj)
        
        print(f1/f0)
        t_res = np.zeros(6)
        t_res[0] = t[0] * f1 / f0
        t_res[1] = t[1] * f1 / f0
        t_res[2] = f1 * t[2] / scale + mn_x_1 
        # t_res[2] = -t[0] * (mn_x_0 * f1 / f0 + f1 / 2) - t[1] * (mn_y_0 * f1 / f0 + f1 / 2) + t[2] * (f1 / (2 * scale)) + f1/2 + mn_x_1
        # t_res[2] = -f1/f0 * (t[0] * mn_x_0 + t[1] * mn_y_0) - f1 * (t[0] + t[1]) + 2*f1*t[2]/scale + mn_x_1 + 1

        t_res[3] = t[3] * f1 / f0
        t_res[4] = t[4] * f1 / f0
        t_res[5] = f1 * t[5] / scale + mn_y_1 
        # t_res[5] = -t[3] * (mn_x_0 * f1 / f0 + f1 / 2) - t[4] * (mn_y_0 * f1 / f0 + f1 / 2) + t[5] * (f1 / (2 * scale)) + f1/2 + mn_y_1
        # t_res[5] = -f1/f0 * (t[3] * mn_x_0 + t[4] * mn_y_0) - f1 * (t[3] + t[4]) + 2*f1*t[5]/scale + mn_y_1 + 1

        return t_res

    @staticmethod
    def poly_to_stroke3(sketches, scale=100.0, step=5, eps=1.5, red_rdp=False, normalize=True, pos_normalize=True, relative=True):
        """
        Attention: this method will normalize the strokes
        convert the given sketches to stroke-3 (x, y, p) format, where x, y are 
         the point relative position to the previous point together with its binary pen state.
         for any two consecutive strokes, a point in the middle will be added with a pen state 1 (pen lifted)
        Args:
            sketches (list): list of sketches to be converted
        
        Returns: the converted sketches to stroke-3 format and store them in arrays
        """
        # TODO: document what step represents 
        # get a copy of sketches to avoid editing on the original sketches
        tmp = []
        for sketch in sketches:
            tmp.append(sketch.get_copy())

        sketches = tmp

        if normalize:
            # find the dimentions of the storkes and reduce
            for sketch in sketches:
                mn_w, mn_h, mx_wh = ObjectUtil.get_normalization_parameters(sketch)
                # normalize strokes
                try:
                    for stroke in sketch.get_strokes():
                        for p in stroke.get_points():
                            if pos_normalize:
                                p.x = ((p.x - mn_w) / mx_wh) * scale
                                p.y = ((p.y - mn_h) / mx_wh) * scale
                            else:    
                                p.x = ((p.x - mn_w) / mx_wh * 2.0 - 1.0) * scale #TODO for segmenentation, use this type of normalization
                                p.y = ((p.y - mn_h) / mx_wh * 2.0 - 1.0) * scale
                except: 
                    print("[ObjectUtil] could not normalize stroke as mx_wh is {0}".format(mx_wh))
                    print("[ObjectUtil] sketch len is:{0}".format(len(sketch)))
                    print(sketch.get_strokes()[0].get_points()[0])
                    print(sketch.get_strokes()[0].get_points()[1])

        
        # reduce using rdp
        if red_rdp:
            sketches = ObjectUtil.reduce_rdp(sketches, epsilon=eps)
        # TODO: using rdp for some small sketches are making the sketch so small that it 
        # is raising an error in the sketchformer  when getting the embeddings.

        converted_sketches = []
        for sketch in sketches:
            strokes_lst = sketch.get_strokes()

            # gather all the points in one stroke
            tmp_stroke_3 = []
            for i in range(len(strokes_lst)):
                if len(strokes_lst[i]) == 1:
                    continue
                
                # add all stroke's points with state 0
                for j, p in enumerate(strokes_lst[i].get_points()):
                    eos = 0 if j < len(strokes_lst[i]) - 1 else 1
                    tmp_stroke_3.append([p.x, p.y, eos])
            
            if relative:
                # get the relative position
                tmp_stroke_3 = np.array(tmp_stroke_3)
                tmp_stroke_3[1:, 0:2] -= tmp_stroke_3[:-1, 0:2]

                # omit the first point
                converted_sketches.append(tmp_stroke_3[1:])
            else:
                converted_sketches.append(tmp_stroke_3)

        return converted_sketches

    @staticmethod 
    def poly_to_accumulative_stroke3(sketches, scale=100.0, step=5, eps=1.5, red_rdp=False, normalize=True):
        # convert to stroke-3
        sketches = ObjectUtil.poly_to_stroke3(sketches, scale=scale, step=step, eps=eps, red_rdp=red_rdp, normalize=normalize, relative=False)
        # accumulate the strokes
        # for i in range(len(sketches)):
        #     sketches[i] = np.insert(sketches[i], 0, np.array((0, 0, 0)), axis=0)
        #     # sketches[i][1:, 0:2] += sketches[i][:-1, 0:2]
        #     sketch = sketches[i]
        #     for i in range(1, len(sketch)):
        #         sketch[i][0:2] += sketch[i-1][0:2]
        
        return sketches

    @staticmethod
    def accumalitive_stroke3_to_poly(sketches):
        # deep copy the arrays
        sketches = copy.deepcopy(sketches)

        # convert to stroke 3
        for sketch in sketches:
            sketch[1:, 0:2] -= sketch[:-1, 0:2]

        # convert from stroke-3 to poly
        return ObjectUtil.stroke3_to_poly(sketches)    

    @staticmethod
    def transform_stroke3(sketch, t):
        tmp = copy.deepcopy(sketch)
        for p in tmp:
            p[0], p[1] = p[0] * t[0] + p[1] * t[1] + t[2], p[0] * t[3] + p[1] * t[4] + t[5] 
        return tmp

    @staticmethod
    def lines_to_strokes(lines, omit_first_point=True):
        """ Convert polyline format to stroke-3 format.
        lines: list of strokes, each stroke has format Nx3 """
        strokes = []
        for line in lines:
            linelen = len(line)
            for i in range(linelen):
                eos = 0 if i < linelen - 1 else 1
                strokes.append([line[i][0], line[i][1], eos])
        strokes = np.array(strokes)
        strokes[1:, 0:2] -= strokes[:-1, 0:2]
        return strokes[1:, :] if omit_first_point else strokes

    @staticmethod
    def get_embedding(objs):
        """extract the sketchformer embedding for the given sketch in polyline format 

        Args:
            sketches ([list]): list of sketches in the conitnuious format of polyline
        """
        # get the pre-trained model
        if ObjectUtil.sketchformer is None:
            ObjectUtil.sketchformer = continuous_embeddings.get_continuous_pretrained_model()

        # obtain the stroke-3 format of the sketches
        objs = ObjectUtil.poly_to_stroke3(objs)

        # return the embeddings
        return ObjectUtil.sketchformer.get_embeddings(objs)

    @staticmethod
    def get_embedding_dist(obj1, obj2):
        embds = ObjectUtil.get_embedding([obj1, obj2])
        return np.linalg.norm(embds[1] - embds[0])


    @staticmethod
    def classify(objs):
        """classify the given sketches of the polyline format 

        Args:
            sketches ([list]): list of sketches in the conitnuious format of polyline
        """
        # get the pre-trained model
        if ObjectUtil.sketchformer is None:
            ObjectUtil.sketchformer = continuous_embeddings.get_continuous_pretrained_model()

        # obtain the stroke-3 format of the sketches
        objs = ObjectUtil.poly_to_stroke3(objs)

        # return the embeddings
        return ObjectUtil.sketchformer.classify(objs)

    @staticmethod
    def reduce_rdp(objs, epsilon=None, mn_len = 5):
        """reduce the given set of UnlabeledObjects using Ramer???Douglas???Peucker algorithm

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
                if epsilon is None:
                    reduced_points = rdp(tmp)
                else:
                    reduced_points = rdp(tmp, epsilon=epsilon)
                
                # obtain the matches points to the reduced points 
                ind, res_points = 0, []
                for pt in stroke.get_points():
                    if ind >= len(reduced_points):
                        break

                    if pt.x == reduced_points[ind][0] and pt.y == reduced_points[ind][1]:
                        res_points.append(pt)
                        ind += 1
                
                if len(res_points) < mn_len:
                    reduced_strokes.append(stroke)
                else:
                    reduced_strokes.append(Stroke(res_points))
            
            reduced_objs.append(UnlabeledObject(reduced_strokes))
            
        return reduced_objs

    def get_centroid(obj):
        """For a given object, return the centroid point

        Args:
            obj (UnlabeledObject): [description]
        
        Return:
            centroid point
        """
        sum_x, sum_y = sum(obj.get_x()), sum(obj.get_y())
        return Point(sum_x / len(obj), sum_y / len(obj))


    cnt = 0
    def extract_objects_from_directory(directory, n_files = -1, re_sampling=0.0, acceptable_labels=None):
        # objs : (N x M) M ordered objects for N sketches
        # labels: (N x M) for each object m in sketch n, store its label
        objs, labels = [], []
        for path in pathlib.Path(directory).iterdir():
            if path.is_dir():
                tmp_objs, tmp_labels = ObjectUtil.extract_objects_from_directory(path, n_files, acceptable_labels=acceptable_labels, re_sampling=re_sampling)
                if n_files != -1:
                    n_files -= len(tmp_objs)
                objs.extend(tmp_objs)
                labels.extend(tmp_labels)

            elif path.is_file():
                if n_files != -1 and len(objs) > n_files:
                    break
                if str(path).endswith(".xml"): 
                    try:
                        # extract all objects in the file along with their labels, and indices
                        tmp_objs, tmp_labels = ObjectUtil.xml_to_UnlabeledObjects(str(path), re_sampling=re_sampling)
                        tmp_objs, tmp_labels = np.asarray(tmp_objs), np.asarray(tmp_labels) 
                        inds = [i for i, lbl in enumerate(tmp_labels) if lbl in acceptable_labels]
                        objs.extend(tmp_objs[inds])
                        labels.extend(tmp_labels[inds])
                        n_files -= len(inds)
                        ObjectUtil.cnt += len(inds)
                    except Exception as e: 
                        print("ObjectUtil] error: could not read file succefully " + str(path))
                        print(str(e))
        return objs, labels

                



    

