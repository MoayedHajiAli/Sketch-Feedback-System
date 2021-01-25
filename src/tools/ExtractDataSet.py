from sketch_object.UnlabeledObject import  UnlabeledObject
from xml.dom import minidom
import os
import pathlib
from xml.dom.minidom import parseString


class ExtractSingleObject:
    """explore all xml files in a directoy, discard from each file all objects, 
    points, and strokes that do not match the given label, and store them in a new directory.
    """
    def __init__(self, labels:list, out_dir:str):
        self.labels, self.out_dir = labels, out_dir
    
    def parse_xml_data(self, file:str):
        """Extract points and strokes for all objects with the given lable

        Args:
            file (str): path to the xml file
            lable (str): label

        Returns:
            list: list of of
        """
        data = minidom.parse(file)
        points = data.getElementsByTagName('point')
        strokes = data.getElementsByTagName('stroke')
        objs = data.getElementsByTagName('Object')

        # extract list of strokes ids for each object with the given label
        matched_objs, matched_strokes, matched_pts = [], [], []
        for obj in objs:
            if str(obj.attributes['Label'].value) not in self.labels:
                continue
            stroke_lst= obj.getElementsByTagName('arg')
            for st in stroke_lst:
                matched_strokes.append(st.firstChild.nodeValue)

            # add to all matched objects
            matched_objs.append(obj.attributes['id'].value)
        
        for st in strokes:
            if st.attributes['id'].value not in matched_strokes:
                continue
            pts = st.getElementsByTagName('arg')
            for pt in pts:
                matched_pts.append(pt.firstChild.nodeValue)
            
        return matched_objs, matched_strokes, matched_pts
    
    def extract_objects(self, filePath):
        matched_objs, matched_strokes, matched_pts = \
            self.parse_xml_data(filePath)
        
        if len(matched_objs) == 0:
            return
        data = minidom.parse(filePath).getElementsByTagName('sketch')[0]
        delete_queue = []
        for child in data.childNodes:
            if type(child) is minidom.Text:
                delete_queue.append(child)
            elif child.tagName == 'point' and child.attributes['id'].value not in matched_pts:
                delete_queue.append(child)
            elif child.tagName == 'stroke' and child.attributes['id'].value not in matched_strokes:
                delete_queue.append(child)
            elif child.tagName == 'Object' and child.attributes['id'].value not in matched_objs:
                delete_queue.append(child)

        for child in delete_queue:
            data.removeChild(child)

        rmv_lines = lambda s : '\n'.join([l for l in s.split('\n') if l.strip() != ''])
        xml_file = rmv_lines(data.toprettyxml())
        file_name = os.path.basename(filePath)
        out_path = os.path.join(self.out_dir, file_name)
        # replace the original file 
        with open(out_path, "w" ) as fs:  
            fs.write(xml_file) 

    def extract(self, dir):
        # explore recursivly all directories
        for path in pathlib.Path(dir).iterdir():
            if path.is_dir():
                self.extract(path)
            elif path.is_file():
                file = os.path.basename(path)
                if file.endswith(".xml"): 
                    try:
                        self.extract_objects(str(path))
                    except Exception as e:
                        print("could not convert file" + file)
    
