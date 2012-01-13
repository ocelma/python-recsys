import sys
import recsys.algorithm
recsys.algorithm.VERBOSE = True
from recsys.utils.svdlibc import SVDLIBC

movielens = sys.argv[1] #ratings.dat movielens file path here
svdlibc = SVDLIBC(movielens)
svdlibc.to_sparse_matrix(sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})
svdlibc.compute()
svd = svdlibc.export()
svdlibc.remove_files()
MOVIEID = 1
print svd.similar(MOVIEID)
print svd.recommend(MOVIEID)
