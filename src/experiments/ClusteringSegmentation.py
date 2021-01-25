import sys
sys.path.insert(0, '../')

from tools.StrokeClustering import DensityClustering
from utils.ObjectUtil import ObjectUtil

dir = "../"
# centroids will be only from the combinations of strokes of obj
cluster = DensityClustering.fromDir(obj, dir, n=None)

# choose the type of clustering
cluster.mut_execlusive_cluster()