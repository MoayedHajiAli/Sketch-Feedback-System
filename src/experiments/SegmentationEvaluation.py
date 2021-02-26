
import sys
sys.path.insert(0, '../')

from tools.ParsingEvaluation import ParsingEvaluation
import os.path as path
import os
#evaluate sketch object level segmentation
test_dir = 'ASIST_Dataset/Data/Data_A/BalanceQuestion'
tmp_dir = '../input_directory/prototypes'
tar_obj_path = 'ASIST_Dataset/Data/Data_A/BoxPointerMergedQuestion/8_2ce97cdb-c300-4b52-a9bf-dc3a1bde7ed7.xml'
# tar_obj_path = 'ASIST_Dataset/Data/Data_A/MoneyQuestion/1_5777f61a-1f9a-45a8-a9aa-7fcd30c8c09a.xml'
test_dir = path.join(path.abspath(path.join(__file__ ,"../../..")), test_dir)
tar_obj_path = path.join(path.abspath(path.join(__file__ ,"../../..")), tar_obj_path)

experiment_id = 8
save_dir = '../results_log/segmentation/iterative/experiment{0}'.format(experiment_id)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
# redirect output to log
sys.stdout = open(os.path.join(save_dir, 'log.out'), 'w+')

evaluator = ParsingEvaluation(str(test_dir), str(tar_obj_path), n_files=-1)
evaluator.evaluate(save_dir=save_dir)

