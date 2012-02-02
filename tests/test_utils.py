import os
from nose import with_setup
from nose.tools import assert_equal, assert_not_equal, assert_raises, assert_true
from operator import itemgetter

import recsys.algorithm
recsys.algorithm.VERBOSE = True

from recsys.utils.svdlibc import SVDLIBC

from tests import MOVIELENS_DATA_PATH


def setup():
    global svdlibc
    svdlibc = SVDLIBC(datafile=os.path.join(MOVIELENS_DATA_PATH, 'ratings.dat'))

def teardown():
    svdlibc.remove_files()

def test_to_sparse_matrix():
    svdlibc.to_sparse_matrix(sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})

def test_compute_svd():
    svdlibc.compute()

def test_export():
    global svd, MOVIEID
    svd = svdlibc.export()
    MOVIEID = 1

def test_similar():
    similars = svd.similar(MOVIEID)
    assert_true(len(similars) == 10)
    assert_true(588 in map(itemgetter(0), similars))

def test_similarity():
    MOVIEID2 = 3114
    assert_equal(round(svd.similarity(MOVIEID, MOVIEID2), 4), round(0.84099896392054219, 4))
