from nose import with_setup
from nose.tools import assert_equal, assert_not_equal, assert_raises, assert_true

from recsys.algorithm.matrix import SparseMatrix as OurSparseMatrix
from divisi2 import make_sparse, SparseMatrix

def setup():
    global matrix, matrix_empty, data

    data = [
	  (4.0, "user1", "item1"),
	  (2.0, "user1", "item3"),
	  (1.0, "user2", "item1"),
	  (5.0, "user2", "item4")]
    matrix = OurSparseMatrix()
    matrix.create(data)

    matrix_empty = OurSparseMatrix()
    matrix_empty.create([])

#def teardown():
#    pass

def test_get_matrix():
    m = matrix.get()
    assert_true(isinstance(m, SparseMatrix))

def test_equal():
    m = make_sparse(data)
    assert_equal(matrix.get(), m)

def test_set_matrix():
    m = make_sparse(data)
    mm = OurSparseMatrix()
    mm.set(m)
    assert_equal(matrix.get(), mm.get())

def test_get_row_len():
    assert_equal(matrix.get_row_len(), 3)

def test_get_col_len():
    assert_equal(matrix.get_col_len(), 2)

def test_get_density():
    assert_equal(matrix.density(), 66.6667)

def test_get_value():
    assert_equal(matrix.value("user1", "item1"), 4.0)

def test_set_value():
    matrix.set_value("user1", "item1", -1.0)
    assert_equal(matrix.value("user1", "item1"), -1.0)

def test_get_value_synonym():
    assert_equal(matrix.value("user1", "item1"), matrix.get_value("user1", "item1"))

# Empty matrix tests
def test_empty_matrix():
    assert_true(matrix_empty.empty())

def test_empty_matrix_equals():
    data = []
    m = OurSparseMatrix()
    m.create(data)
    assert_equal(m.get(), matrix_empty.get())

def test_empty_matrix_get():
    assert_true(isinstance(matrix_empty.get(), SparseMatrix))

def test_empty_matrix_get_value():
    assert_raises(ValueError, matrix_empty.get_value, 1, 1)

def test_empty_matrix_set_value():
    assert_raises(ValueError, matrix_empty.set_value, 1, 1, 5.0)

def test_empty_matrix_get_row():
    assert_raises(ValueError, matrix_empty.get_row, 1)

def test_empty_matrix_get_col():
    assert_raises(ValueError, matrix_empty.get_col, 1)

def test_empty_matrix_get_row_len():
    assert_raises(ValueError, matrix_empty.get_row_len)

def test_empty_matrix_get_col_len():
    assert_raises(ValueError, matrix_empty.get_col_len)
 
