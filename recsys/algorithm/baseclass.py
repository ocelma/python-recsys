"""
.. module:: algorithm
   :synopsis: Base class Algorithm

.. moduleauthor:: Oscar Celma <ocelma@bmat.com>

"""
import sys
from scipy.cluster.vq import kmeans2 #for kmeans method
from random import randint #for kmeans++ (_kinit method)
#from scipy.linalg import norm #for kmeans++ (_kinit method)
from scipy import array #for kmeans method
from numpy import sum
from numpy.linalg import norm #for _cosine and kmeans++ (_kinit method)
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
        self._matrix_similarity = None #self-similarity matrix (only for the input Matrix rows)
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
        if not self.get_matrix_similarity() or self.get_matrix_similarity().get() is None:
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
        if not self.get_matrix_similarity() or self.get_matrix_similarity().get() is None:
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
        if not self.get_matrix_similarity() or self.get_matrix_similarity().get() is None:
            self.compute()
        return self.get_matrix_similarity().value(i, j)

    def predict(self, i, j, MIN_VALUE=None, MAX_VALUE=None):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def recommend(self, i, n=10):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    ### OTHER METHODS ###
    def _cosine(self, v1, v2):
        return float(divisi2.dot(v1,v2) / (norm(v1) * norm(v2)))

    def centroid(self, ids, are_rows=True):
        if VERBOSE:
            sys.stdout.write('Computing centroid for ids=%s\n' % str(ids))
        points = []
        for id in ids:
            if are_rows:
                point = self.get_matrix().get_row(id)
            else:
                point = self.get_matrix().get_col(id)
            points.append(point)
        M = divisi2.SparseMatrix(points)
        return M.col_op(sum)/len(points) #TODO numpy.sum seems slower?

    def _kinit(self, X, k):
        #Init k seeds according to kmeans++
        n = X.shape[0]
        #Choose the 1st seed randomly, and store D(x)^2 in D[]
        centers = [X[randint(0, n-1)]]
        D = [norm(x-centers[0])**2 for x in X]

        for _ in range(k-1):
            bestDsum = bestIdx = -1
            for i in range(n):
                #Dsum = sum_{x in X} min(D(x)^2,||x-xi||^2)
                Dsum = reduce(lambda x,y:x+y,
                              (min(D[j], norm(X[j]-X[i])**2) for j in xrange(n)))
                if bestDsum < 0 or Dsum < bestDsum:
                    bestDsum, bestIdx = Dsum, i
            centers.append(X[bestIdx])
            D = [min(D[i], norm(X[i]-X[bestIdx])**2) for i in xrange(n)]
        return array(centers)

    def kmeans(self, id, k=5, is_row=True):
        """
        K-means clustering. http://en.wikipedia.org/wiki/K-means_clustering

        Clusterizes the (cols) values of a given row, or viceversa

        :param id: row (or col) id to cluster its values
        :param k: number of clusters
        :param is_row: is param *id* a row (or a col)?
        :type is_row: Boolean
        """
        # TODO: switch to Pycluster?
        # http://pypi.python.org/pypi/Pycluster
        if VERBOSE:
            sys.stdout.write('Computing k-means, k=%s, for id %s\n' % (k, id))
        point = None
        if is_row:
            point = self.get_matrix().get_row(id)
        else:
            point = self.get_matrix().get_col(id)
        points = []
        points_id = []
        for i in point.nonzero_entries():
            label = point.label(i)
            points_id.append(label)
            if not is_row:
                points.append(self.get_matrix().get_row(label))
            else:
                points.append(self.get_matrix().get_col(label))
        #return kmeans(array(points), k)
        if VERBOSE:
            sys.stdout.write('id %s has %s points\n' % (id, len(points)))
        M = array(points)

        MAX_POINTS = 150
        # Only apply Matrix initialization if num. points is not that big!
        if len(points) <= MAX_POINTS:
            centers = self._kinit(array(points), k)
            centroids, labels = kmeans2(M, centers, minit='matrix')
        else:
            centroids, labels = kmeans2(M, k, minit='random')
        i = 0
        clusters = dict()
        for cluster in labels:
            if not clusters.has_key(cluster): 
                clusters[cluster] = dict()
                clusters[cluster]['centroid'] = centroids[cluster]
                clusters[cluster]['points'] = []
            clusters[cluster]['points'].append(points_id[i])
            i += 1
        return clusters

