import os
from nose import with_setup
from nose.tools import assert_equal, assert_not_equal, assert_raises, assert_true

from recsys.datamodel.user import User
from recsys.datamodel.item import Item
from recsys.datamodel.data import Data

from tests import MOVIELENS_DATA_PATH

USERID = 'ocelma'
MOVIEID = 1
ARTISTID = 'u2'
PLAYS = 25

def setup():
    global user, items, data
    user = User(USERID)
    items = _read_items(os.path.join(MOVIELENS_DATA_PATH, 'movies.dat'))
    data = Data()
    data.load(os.path.join(MOVIELENS_DATA_PATH, 'ratings.dat'), sep='::', format={'col':0, 'row':1, 'value':2, 'ids':int})

def teardown():
    pass

# Read movie info
def _read_items(filename):
    items = dict()
    for line in open(filename):
        #1::Toy Story (1995)::Animation|Children's|Comedy
        data =  line.strip('\r\n').split('::')
        item_id = data[0]
        item_name = data[1]
        genres = data[2:]
        items[item_id] = Item(item_id)
        items[item_id].add_data({'name': item_name, 'genres': genres})
    return items

# DATA tests
def test_data_split_train_test():
    train, test = data.split_train_test()
    assert_equal(len(train), 800167)
    assert_equal(len(test), 200042)

def test_data_extend():
    dataset = [(1,2,3), (4,5,6)]
    dataset2 = [(7,8,9), (10,11,12)]
    data = Data()
    data.set(dataset)
    assert_equal(len(data), 2)

    data.set(dataset2, extend=True)
    assert_equal(len(data), 4)

def test_data_add_tuple():
    VALUE = 4.0
    tuple = (VALUE, 'row_id', 'col_id')
    data = Data()
    data.add_tuple(tuple)
    assert_equal(data[0][0], VALUE)

def test_data_add_tuple_error_format():
    tuple = (4, 'row_id', 'col_id', 'another error value!')
    assert_raises(ValueError, data.add_tuple, tuple)

def test_data_add_tuple_error_format2():
    tuple = ('row_id', 'col_id')
    assert_raises(ValueError, data.add_tuple, tuple)

def test_data_add_tuple_value_none():
    tuple = (None, 'row_id', 'col_id')
    assert_raises(ValueError, data.add_tuple, tuple)

def test_data_add_tuple_value_empty_string():
    tuple = ('', 'row_id', 'col_id')
    assert_raises(ValueError, data.add_tuple, tuple)

def test_data_add_tuple_value_string():
    tuple = ('1.0', 'row_id', 'col_id')
    assert_raises(ValueError, data.add_tuple, tuple)

#def test_data_add_tuple_zero_value():
#    tuple = (0, 'row_id', 'col_id')
#    assert_raises(ValueError, data.add_tuple, tuple)

def test_data_add_tuple_row_id_empty():
    tuple = (1, '', 'col_id')
    assert_raises(ValueError, data.add_tuple, tuple)

def test_data_add_tuple_col_id_empty():
    tuple = (1, 'row_id', '')
    assert_raises(ValueError, data.add_tuple, tuple)

def test_data_add_tuple_row_id_none():
    tuple = (1, None, 'col_id')
    assert_raises(ValueError, data.add_tuple, tuple)

def test_data_add_tuple_col_id_none():
    tuple = (1, 'row_id', None)
    assert_raises(ValueError, data.add_tuple, tuple)

#USER tests
def test_user_build():
    u = User(USERID)
    assert_equal(u.get_id(), USERID)

def test_user_add_item():
    u = User(USERID)
    item = Item(ARTISTID)
    item.add_data({'name': ARTISTID})
    u.add_item(item, PLAYS)
    assert_equal(str(u.get_items()), '[(u2, 25)]')

def test_user_get_items():
    for item, weight in user.get_items():
        assert_true(isinstance(item, Item))

#ITEM tests
def test_item_build():
    data = dict()
    data['name'] = 'u2'
    data['popularity'] = 5.0
    item = Item(MOVIEID)
    item.add_data(data)
    assert_true(isinstance(item, Item))
    assert_equal(str(item.get_data()), "{'popularity': 5.0, 'name': 'u2'}")

def test_item_get_data():
    item = items['1']
    assert_true(isinstance(item, Item))
    assert_equal(str(item.get_data()), "{\'genres\': [\"Animation|Children\'s|Comedy\"], \'name\': \'Toy Story (1995)\'}")

