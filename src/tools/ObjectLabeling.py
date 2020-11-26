from scipy.io import loadmat
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import uuid
from xml.dom import minidom
import numpy as np
import scipy.io
import os
import pathlib
from xml.dom.minidom import parseString


class ObjectLabeling:
    """Go throw all the xml files and label the objects using the matlab files
    """
    def mat_data(self, file):
        data = loadmat(file)['annotData'][0][0]
        label = str(data[1][0])
        data = data[0][0]
        coords, t = [], []
        for st in data:
            coords.extend(st[0])
            t.extend(st[1])
        points = []
        for p, t in zip(coords, t):
            points.append([p[0], p[1], t[0]])
        return points, label

    def xml_data(self, file):
        data = minidom.parse(file)
        points = data.getElementsByTagName('point')
        strokes = data.getElementsByTagName('stroke')
        
        # creat dictionary for id -> Point
        pt_id, id_point, id_stroke = {}, {}, {}
        for el in points:
            x = int(el.attributes['x'].value)
            y = int(el.attributes['y'].value)
            time = float(el.attributes['time'].value)
            pid = el.attributes['id'].value
            id_point[pid] = (x ,y , time)
            pt_id[repr([x, y])] = pid

        # extract strokes
        for st in strokes:
            sid = st.attributes['id'].value
            pts = st.getElementsByTagName('arg')
            for pt in pts:
                pid = pt.firstChild.nodeValue
                id_stroke[pid] = sid
        
        return pt_id, id_point, id_stroke 
    
    def collect_strokes(self, directory, filePath):
        pt_id, id_point, id_stroke = self.xml_data(filePath)
        lookup = os.path.basename(filePath)[:-4]
        root = minidom.parse(filePath)
        for path in pathlib.Path(directory).iterdir():
            if path.is_file():
                fileName = os.path.basename(path)
                if(fileName.startswith("annot_" + lookup)):
                    points, label = self.mat_data(path)
                    ids = set()
                    for p in points:
                        pt = repr([p[0], p[1]])
                        ids.add(id_stroke[pt_id[pt]]) 
                    obj = root.createElement("Object")
                    obj.setAttribute("Label", label)
                    obj.setAttribute("id", str(uuid.uuid4()))
                    for sid in ids:
                        st = root.createElement('arg')
                        st.setAttribute("type", "Stroke")
                        st.appendChild(root.createTextNode(sid))
                        obj.appendChild(st)
                    root.firstChild.appendChild(obj)

        rmv_lines = lambda s : '\n'.join([l for l in s.split('\n') if l.strip() != ''])
        xml_file = rmv_lines(root.toprettyxml())
        # replace the original file 
        with open(filePath, "w" ) as fs:  
            fs.write(xml_file) 

    def explore(self, directory):
        for path in pathlib.Path(directory).iterdir():
            if path.is_dir():
                self.explore(path)
            elif path.is_file():
                file = os.path.basename(path)
                if file.endswith(".xml"): 
                    try:
                        self.collect_strokes(directory, str(path))
                    except:
                        print("could not convert file" + file)

data_dir = '../ASIST_Dataset/Data/Data_A'
# ObjectLabeling().explore(data_dir)


