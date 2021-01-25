import sys
sys.path.insert(0, "../")

from tools.ExtractDataSet import ExtractSingleObject
from utils.ObjectUtil import ObjectUtil
import os.path as path
import os

INP_DIR = 'ASIST_Dataset/Data/Data_A'
INP_DIR = path.join(path.abspath(path.join(__file__, "../../..")), INP_DIR)
OUT_DIR = 'ASIST_Dataset/Data/Data_B/Triangles'
OUT_DIR = path.join(path.abspath(path.join(__file__, "../../..")), OUT_DIR)

if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

# objs, _ = ObjectUtil.xml_to_UnlabeledObjects(OUT_DIR)
# # objs[0].visualize()
# objs[1].visualize()
circle_ds = ExtractSingleObject(['Triangle', 'Upsidedown Triangle'], OUT_DIR)
circle_ds.extract(INP_DIR)