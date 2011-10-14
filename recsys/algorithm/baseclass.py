"""
.. module:: algorithm
   :synopsis: Base class Algorithm

.. moduleauthor:: Oscar Celma <ocelma@bmat.com>

"""
import sys
#from numpy import sum
from numpy.linalg import norm
from scipy.cluster.vq import kmeans#, kmeans2 #for kmeans method
from scipy import array #for kmeans method
try:
    import divisi2
except:
    from csc import divisi2

from recsys.algorithm import VERBOSE
from recsys.algorithm.matrix import SparseMatrix
from recsys.datamodel.data import Data

class Algorithm(object):
    """
    Base class Algorithm

    It has the basic methods to load a dataset, get the matrix and the raw input
    data, add more data (tuples), etc.

    Any other Algorithm derives from this base class
    """
    def __init__(self):
        self._data = Data()
        self._matrix = SparseMatrix()
        self._matrix_similarity = None #self-similarity matrix (for either the rows or the cols of the input Matrix)
        self._matrix_and_data_aligned = False #both Matrix and Data contain the same info?

    def __len__(self):
        return len(self.get_data())

    def __repr__(self):
        s = '%d rows.' % len(self.get_data())
        if len(self.get_data()):
            s += '\nE.g: %s' % str(self.get_data()[0])
        return s

    def get_matrix(self):
        """
        :returns: matrix *M*
        """
        if not self._matrix.get():
            self.create_matrix()
        return self._matrix

    def get_matrix_similarity(self):
        """
        :returns: the self-similarity matrix
        """
        return self._matrix_similarity

    def set_data(self, data):
        """
        Sets the raw dataset (input for matrix *M*)

        :param data: a Dataset class (list of tuples <value, row, col>)
        :type data: Data
        """
        #self._data = Data()
        #self._data.set(data)
        self._data = data
        self._matrix_and_data_aligned = False

    def get_data(self):
        """
        :returns: An instance of Data class. The raw dataset (input for matrix *M*). 
        """
        return self._data

    def add_tuple(self, tuple):
        """
        Add a tuple in the dataset

        :param tuple: a tuple containing <rating, user, item> information. Or, more general: <value, row, col>
        """
        self.get_data().add_tuple(tuple)
        self._matrix_and_data_aligned = False

    def load_data(self, filename, force=True, sep='\t', format={'value':0, 'row':1, 'col':2}, pickle=False):
        """
        Loads a dataset file

        See params definition in *datamodel.Data.load()*
        """
        if force:
            self._data = Data()
            self._matrix_similarity = None

        self._data.load(filename, force, sep, format, pickle)
    
    def save_data(self, filename, pickle=False):
        """
        Saves the dataset in divisi2 matrix format (i.e: value <tab> row <tab> col)

        :param filename: file to store the data
        :type filename: string
        :param pickle: save in pickle format?
        :type filename: boolean
        """
        self._data.save(filename, pickle)

    def create_matrix(self):
        if VERBOSE:
            sys.stdout.write('Creating matrix (%s tuples)\n' % len(self._data))
        try:
            self._matrix.create(self._data.get())
        except AttributeError:
            self._matrix.create(self._data)

        if VERBOSE:
            sys.stdout.write("Matrix density is: %s%%\n" % self._matrix.density())
        self._matrix_and_data_aligned = True

    def compute(self, min_values=None):
        if self._matrix.empty() and (not isinstance(self._data, list) and not self._data.get()):
            raise ValueError('No data set. Matrix is empty!')
        if self._matrix.empty() and (isinstance(self._data, list) and not self._data):
            raise ValueError('No data set. Matrix is empty!')
        if not self._matrix.empty() or not self._matrix_and_data_aligned:
            self.create_matrix()

        if min_values:
            if VERBOSE:
                sys.stdout.write('Updating matrix: squish to at least %s values\n' % min_values)
            self._matrix.set(self._matrix.get().squish(min_values))

    def _get_row_similarity(self, i):
        if not self.get_matrix_similarity():
            self.compute()
        try:
            return self.get_matrix_similarity().get_row(i)
        except KeyError:
            raise KeyError("%s not found!" % i)

    def similar(self, i, n=10):
        """
        :param i: a row in *M*
        :type i: user or item id
        :param n: number of similar elements
        :type n: int
        :returns: the most similar elements of *i*
        """
        if not self.get_matrix_similarity():
            self.compute()
        return self._get_row_similarity(i).top_items(n)

    def similarity(self, i, j):
        """
        :param i: a row in *M*
        :type i: user or item id
        :param j: a row in *M*
        :type j: user or item id
        :returns: the similarity between the two elements *i* and *j*
        """
        if not self.get_matrix_similarity():
            self.compute()
        return self.get_matrix_similarity().value(i, j)

    def predict(self, i, j, MIN_VALUE=None, MAX_VALUE=None):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def recommend(self, i, n=10):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    ### OTHER METHODS ###
    def _cosine(self, v1, v2):
        return float(divisi2.dot(v1,v2) / (norm(v1) * norm(v2)))

    def centroid(self, ids, is_row=True):
        if VERBOSE:
            sys.stdout.write('Computing centroid for ids=%s\n' % str(ids))
        points = []
        for id in ids:
            if is_row:
                point = self.get_matrix().get_row(id)
            else:
                point = self.get_matrix().get_col(id)
            points.append(point)
        M = divisi2.SparseMatrix(points)
        return M.col_op(sum)/len(points) #TODO numpy.sum?

    def kmeans(self, id, k=5, is_row=True):
        if VERBOSE:
            sys.stdout.write('Computing k-means, with k=%s\n' % k)
        point = None
        if is_row:
            point = self.get_matrix().get_row(id)
        else:
            point = self.get_matrix().get_col(id)
        points = []
        for i in point.nonzero_entries():
            label = point.label(i)
            if not is_row:
                points.append(self.get_matrix().get_row(label))
            else:
                points.append(self.get_matrix().get_col(label))
        return kmeans(array(points), k)
