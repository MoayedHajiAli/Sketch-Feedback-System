import sys
sys.path.insert(0, '../')

from tools.StrokeClustering import DensityClustering

cluster = DensityClustering.fromDir(obj, dir, n=2)

# choose the type of clustering
cluster.mut_execlusive_cluster()