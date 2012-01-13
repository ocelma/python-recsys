import os
from nose import with_setup
from nose.tools import assert_equal, assert_not_equal, assert_raises, assert_true

from recsys.utils.svdlibc import SVDLIBC

from tests import MOVIELENS_DATA_PATH


def setup():
    global svdlibc
    svdlibc = SVDLIBC(os.path.join(MOVIELENS_DATA_PATH, 'ratings.dat'))

def teardown():
    svdlibc.remove_files()

def test_to_sparse_matrix():
    svdlibc.to_sparse_matrix(sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})

def test_compute_svd():
    svdlibc.compute()

def test_export():
    svd = svdlibc.export()
    MOVIEID = 1
    assert_true(len(svd.similar(MOVIEID)) == 10)
