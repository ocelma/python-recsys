# -*- coding: utf-8 -*-
import os
import codecs

from random import randrange
from nose import with_setup
from nose.tools import assert_equal, assert_not_equal, assert_raises, assert_true

import recsys.algorithm
recsys.algorithm.VERBOSE = True

from recsys.datamodel.data import Data
from recsys.datamodel.user import User
from recsys.datamodel.item import Item
from recsys.algorithm.factorize import SVD

from tests import MOVIELENS_DATA_PATH

FACTORS = 50 #The K for SVD

#Define some global vars.
ITEMID = 1
USERID1 = 1
USERID2 = 5
NUM_SIMILARS = 10

MIN_RATING = 1.0
MAX_RATING = 5.0

svd = None
users = None
items = None

def setup():
    global users, items, svd

    print 'Reading items...'
    items = _read_items(os.path.join(MOVIELENS_DATA_PATH, 'movies.dat'))
    users = []

    svd = SVD()
    svd.load_data(filename=os.path.join(MOVIELENS_DATA_PATH, 'ratings.dat'), sep='::', format={'col':0, 'row':1, 'value':2, 'ids':int})

def teardown():
    pass

# Read movie info
def _read_items(filename):
    items = dict()
    for line in codecs.open(filename, 'r', 'latin1'):
        #1::Toy Story (1995)::Animation|Children's|Comedy
        data =  line.strip('\r\n').split('::')
        item_id = int(data[0])
        item_name = data[1]
        str_genres = data[2]
        genres = []
        for genre in str_genres.split('|'):
            genres.append(genre)
        items[item_id] = Item(item_id)
        items[item_id].add_data({'name': item_name, 'genres': genres})
    return items

### TESTS ###
def test_matrix_get_row_len():
    row_len = 6040
    assert_equal(svd.get_matrix().get_row_len(), row_len)

def test_matrix_get_col_len():
    col_len = 3706
    assert_equal(svd.get_matrix().get_col_len(), col_len)

def test_matrix_density():
    density = 4.4684
    assert_equal(svd.get_matrix().density(), density)

def test_get_data():
    num_rows = 1000209
    assert_equal(len(svd.get_data()), num_rows)

def test_save_data():
    data_in = svd.get_data()
    svd.save_data(os.path.join(MOVIELENS_DATA_PATH, 'ratings.matrix.saved'))

    svd.load_data(os.path.join(MOVIELENS_DATA_PATH, 'ratings.matrix.saved'))
    data_saved = svd.get_data()

    assert_equal(len(data_in), len(data_saved))
    assert_true(isinstance(data_saved, Data))

def test_utf8_data():
    data_in = Data()

    NUM_PLAYS = 69
    ITEMID = u'Bj\xf6rk' 
    data_in.add_tuple([NUM_PLAYS, ITEMID, USERID1])

    NUM_PLAYS = 34
    ITEMID = 'Björk' 
    data_in.add_tuple([NUM_PLAYS, ITEMID, USERID2])

    data_in.save(os.path.join(MOVIELENS_DATA_PATH, 'ratings.matrix.saved.utf8'))

    data_saved = Data()
    data_saved.load(os.path.join(MOVIELENS_DATA_PATH, 'ratings.matrix.saved.utf8'))

    assert_equal(len(data_in), len(data_saved))

def test_save_pickle():
    data_in = svd.get_data()
    svd.save_data(os.path.join(MOVIELENS_DATA_PATH, 'ratings.matrix.pickle'), pickle=True)

    svd.load_data(os.path.join(MOVIELENS_DATA_PATH, 'ratings.matrix.pickle'), pickle=True)
    data_saved = svd.get_data()

    assert_equal(len(data_in), len(data_saved))
    assert_true(isinstance(data_saved, Data))

def test_load_pickle():
    svd = SVD()
    svd.load_data(os.path.join(MOVIELENS_DATA_PATH, 'ratings.matrix.pickle'), pickle=True)
    assert_true(isinstance(svd.get_data(), Data))

def test_compute_empty_matrix():
    svd = SVD()
    assert_raises(ValueError, svd.compute)

def test_compute_svd():
    svd.load_data(filename=os.path.join(MOVIELENS_DATA_PATH, 'ratings.dat'), sep='::', format={'col':0, 'row':1, 'value':2, 'ids':int})
    svd.compute(FACTORS, min_values=10, pre_normalize=None, mean_center=True, post_normalize=True)

    user_ids = [1, 2, 3]
    item_ids = [1, 48, 3114, 25, 44, 3142, 617]
    print
    for item_id in item_ids:
        item_name = items[item_id].get_data()['name']
        #Item similarity (in item_ids)
        for other_item_id in item_ids:
            other_item_name = items[other_item_id].get_data()['name']
            print item_name, other_item_name, svd.similarity(item_id, other_item_id)
        print
        #Similar items for item_id
        for sim_item, similarity in svd.similar(item_id, NUM_SIMILARS):
            print item_name, items[sim_item].get_data()['name'], similarity
        print
        #Predicted ratings for (user_id, item_id)
        for user_id in user_ids:
            pred_rating = svd.predict(user_id, item_id, MIN_RATING, MAX_RATING)
            print item_name, user_id, svd.get_matrix().value(item_id, user_id), pred_rating
        print

def test_save_model():
    svd.save_model(os.path.join(MOVIELENS_DATA_PATH, 'SVD_matrix'), options={'k': FACTORS})

def test_load_model():
    svd2 = SVD()
    svd2.load_model(os.path.join(MOVIELENS_DATA_PATH, 'SVD_matrix'))
    recs_svd = svd.recommend(USERID1, NUM_SIMILARS, is_row=False)
    recs_svd2 = svd2.recommend(USERID1, NUM_SIMILARS, is_row=False)
    assert_equal(recs_svd, recs_svd2)

def test_recommendations():
    # Recommendations are based on svd.compute() from test_compute_svd()
    print
    for item_id, relevance in svd.recommend(USERID1, NUM_SIMILARS, is_row=False): # Recommend items to USERID1
        print USERID1, items[item_id].get_data()['name'], relevance
    print
    for item_id, relevance in svd.recommend(USERID2, NUM_SIMILARS, only_unknowns=False, is_row=False): # Recommend (unknown) items to USERID2
        print USERID2, items[item_id].get_data()['name'], relevance

def test_predictions():
    # Predictions are based on previously svd.compute() from test_compute_svd(). That is:
    # svd.compute(FACTORS, min_values=10, pre_normalize=None, mean_center=True, post_normalize=True)
    print
    item_ids = [1, 48, 3114]
    for item_id in item_ids:
        print item_id, USERID1
        print 'real value     ', svd.get_matrix().value(item_id, USERID1)
        print 'predicted value', svd.predict(item_id, USERID1, MIN_RATING, MAX_RATING)

def test_centroid():
    item_ids = [1, 48, 3114]
    centroid = svd.centroid(item_ids)

    item1 = svd._U.row_named(item_ids[0])
    assert_true(svd._cosine(centroid, item1) > 0.85)
    item2 = svd._U.row_named(item_ids[1])
    assert_true(svd._cosine(centroid, item2) > 0.10)
    item3 = svd._U.row_named(item_ids[2])
    assert_true(svd._cosine(centroid, item3) > 0.80)

def test_largest_eigenvectors():
    #Look up the first column of svd._U, and ask for its top items.
    item_ids = svd._U[:,0].top_items(5)
    for item_id, relevance in item_ids:
        print item_id, items[item_id].get_data()['name'], relevance
    print
#    item_ids = (-svd._U[:,0]).top_items(5)
#    for item_id, relevance in item_ids:
#        print items[item_id].get_data()['name'], relevance
#    user_ids, relevance = svd._V[:,0].top_items(5)
#    for user_id in user_ids:
#        print user_id

def test_kmeans_kinit():
    k = 5
    col = svd._V.row_named(USERID1)
    print svd._kinit(col, k)

def test_kmeans():
    item_ids = [1, 48, 3114, 25, 44, 3142, 617, 1193, 3408]
    # K-means based on a list of ids
    clusters = svd.kmeans(item_ids, k=3, are_rows=True)
    #print clusters
    for cluster in clusters.values():
        for other_cluster in clusters.values():
            print svd._cosine(cluster['centroid'], other_cluster['centroid'])
        print
    clusters = svd.kmeans(USERID1, are_rows=False)
    print clusters

def test_add_tuple():
    #This test goes in the end. Else it destroys the other tests! (as it adds one more row to the original matrix)
    num_rows = len(svd.get_data())
    svd.add_tuple((5.0, USERID1, items[ITEMID].get_id()))
    assert_equal(len(svd.get_data()), num_rows+1)

