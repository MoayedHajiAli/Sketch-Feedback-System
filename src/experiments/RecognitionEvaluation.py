import sys
sys.path.insert(0, '../')
from tools.ClassEvaluation import ClassEvaluation

eval = ClassEvaluation([], [], re_sampling=0.5)

# add prototype samples
eval.add_file('../input_directory/prototypes/p1.xml')
eval.add_file('../input_directory/prototypes/p2.xml')
eval.add_file('../input_directory/prototypes/p3.xml')
eval.add_file('../input_directory/prototypes/p4.xml')

print("Labels: ", eval.labels)
eval.start('.../ASIST_Dataset/Data/Data_A', 100)