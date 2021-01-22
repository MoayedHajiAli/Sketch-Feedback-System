
import sys
sys.path.insert(0, '../')

from tools.ParsingEvaluation import ParsingEvaluation
import os.path as path
#evaluate sketch object level segmentation
test_dir = 'ASIST_Dataset/Data/Data_A/MoneyQuestion'
tar_obj_path = 'ASIST_Dataset/Data/Data_A/MoneyQuestion/1_5777f61a-1f9a-45a8-a9aa-7fcd30c8c09a.xml'
test_dir = path.join(path.abspath(path.join(__file__ ,"../../..")), test_dir)
tar_obj_path = path.join(path.abspath(path.join(__file__ ,"../../..")), tar_obj_path)

evaluator = ParsingEvaluation(str(test_dir), str(tar_obj_path), n_files=3)
evaluator.evaluate()

