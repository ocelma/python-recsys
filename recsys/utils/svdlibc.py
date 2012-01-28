import sys
import codecs
import os
from operator import itemgetter
import csv
from numpy import array
from divisi2.ordered_set import OrderedSet
from csc.divisi2.dense import DenseMatrix
from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data
from recsys.algorithm import VERBOSE

# Path to 'svd' executable [ http://tedlab.mit.edu/~dr/SVDLIBC/ ]
PATH_SVDLIBC = '/usr/local/bin/'

class SVDLIBC(object):
    def __init__(self, datafile=None, matrix='matrix.dat', prefix='svd'):
        self._data_file = datafile
        self._matrix_file = matrix
        self._svd_prefix = prefix

    def compute(self, k=100, matrix=None, prefix=None):
        if matrix:
            self._matrix_file = matrix
        if prefix:
            self._svd_prefix = prefix
        if VERBOSE:
            sys.stdout.write('SVDLIBC: Computing svd(k=%s) from %s, saving it to %s\n' % (k, self._matrix_file, self._svd_prefix))
        error_code = os.spawnv(os.P_WAIT, PATH_SVDLIBC + 'svd', ['-r st', '-d%d' % k, '-o%s' % self._svd_prefix, self._matrix_file])
        if error_code == 127:
            raise IOError('svd executable not found in: %s. You might need to download it: %s' 
                    % (PATH_SVDLIBC + 'svd', 'http://tedlab.mit.edu/~dr/SVDLIBC/'))

    def set_matrix(self, matrix):
        self._matrix_file = matrix

    def remove_files(self):
        PREFIX = self._svd_prefix
        file_Ut = PREFIX + '-Ut'
        file_Vt = PREFIX + '-Vt'
        file_S = PREFIX + '-S'
        file_row_ids = '%s.ids.rows' % self._svd_prefix
        file_col_ids = '%s.ids.cols' % self._svd_prefix

        files = [self._matrix_file, file_Ut, file_Vt, file_S, file_row_ids, file_col_ids]
        for file in files:
            if not os.path.exists(file):
                raise IOError('could not delete file %s' % file)
            os.remove(file)

    def to_sparse_matrix(self, sep='\t', format=None):
        # http://tedlab.mit.edu/~dr/SVDLIBC/SVD_F_ST.html
        data = Data()
        data.load(self._data_file, sep=sep, format=format)

        f = open(self._matrix_file, 'w')
        f_row_ids = codecs.open('%s.ids.rows' % self._svd_prefix, 'w', 'utf8')
        f_col_ids = codecs.open('%s.ids.cols' % self._svd_prefix, 'w', 'utf8')

        num_rows = len(set(map(itemgetter(1), data)))
        num_cols = len(set(map(itemgetter(2), data)))
        non_zero = len(data)
        f.write('%s %s %s\n' % (num_rows, num_cols, non_zero))

        #print 'sorting data by col'
        l = data.get()
        #l.sort(key=itemgetter(2, 1)) #by col, and then row
        l.sort(key=itemgetter(2))

        rows = dict()
        cols = dict()
        prev_col_id = None
        col_values = []
        row, col = (0, 0)
        for value, row_id, col_id in l:
            #if not row_id or not col_id or not value:
            #    if VERBOSE:
            #        sys.stdout.write('Skipping: %s, %s, %s\n' % (value, row_id, col_id))
            #    continue
            if col_id != prev_col_id:
                if col_values:
                    f.write('%s\n' % len(col_values))
                    for col_row_id, col_value in col_values:
                        _row = rows[col_row_id]
                        f.write('%s %s\n' % (_row, col_value))
                col_values = []
                cols[col_id] = col
                col += 1
            if not rows.has_key(row_id):
                rows[row_id] = row
                row += 1
            col_values.append((row_id, value))
            prev_col_id = col_id
        if col_values:
            f.write('%s\n' % len(col_values))
            for col_row_id, col_value in col_values:
                row = rows[col_row_id]
                f.write('%s %s\n' % (row, col_value))
            cols[col_id] = col
        f.close()

        # Now write f_row_ids and f_col_ids
        rows = rows.items()
        rows.sort(key=itemgetter(1))
        for row_id, _ in rows:
            if row_id == '':
                continue
            if isinstance(row_id, int):
                row_id = str(row_id)
            f_row_ids.write(row_id + '\n')
        f_row_ids.close()

        cols = cols.items()
        cols.sort(key=itemgetter(1))
        for col_id, _ in cols:
            if col_id == '':
                continue
            if isinstance(col_id, int):
                col_id = str(col_id)
            f_col_ids.write(col_id + '\n')
        f_col_ids.close()

    def export(self):
        # http://tedlab.mit.edu/~dr/SVDLIBC/SVD_F_DT.html
        # only importing default 'dt' S, Ut and Vt (dense text output matrices)
        PREFIX = self._svd_prefix
        file_Ut = PREFIX + '-Ut'
        file_Vt = PREFIX + '-Vt'
        file_S = PREFIX + '-S'
        # Not really used:
        file_U = PREFIX + '-U'
        file_V = PREFIX + '-V'
        
        # Read matrices files (U, S, Vt), using CSV (it's much faster than numpy.loadtxt()!)
        if VERBOSE:
            sys.stdout.write('Reading files: %s, %s, %s\n' % (file_Ut, file_Vt, file_S))
        try:
            Ut = array(list(csv.reader(open(file_Ut),delimiter=' '))[1:]).astype('float')
            U = Ut.transpose()
        except:
            U = array(list(csv.reader(open(file_U),delimiter=' '))[1:]).astype('float')
        try:
            Vt = array(list(csv.reader(open(file_Vt),delimiter=' '))[1:]).astype('float')
            V = Vt.transpose()
        except:
            V = array(list(csv.reader(open(file_V),delimiter=' '))[1:]).astype('float')
            #Vt = V.transpose()
        _S = array(list(csv.reader(open(file_S),delimiter=' '))[1:]).astype('float')
        S = _S.reshape(_S.shape[0], )
        
        PREFIX_INDEXES = PREFIX + '.ids.'
        file_U_idx = PREFIX_INDEXES + 'rows'
        file_V_idx = PREFIX_INDEXES + 'cols'
        if VERBOSE:
            sys.stdout.write('Reading index files: %s, %s\n' % (file_U_idx, file_V_idx))
        try:
            U_idx = [ int(idx.strip()) for idx in open(file_U_idx)]
        except:
            U_idx = [ idx.strip() for idx in open(file_U_idx)]
        try:
            V_idx = [ int(idx.strip()) for idx in open(file_V_idx)]
        except:
            V_idx = [ idx.strip() for idx in open(file_V_idx)]
        
        #Check no duplicated IDs!!!
        assert(len(U_idx) == len(OrderedSet(U_idx))), 'There are duplicated row IDs!'
        assert(len(U_idx) == U.shape[0]), 'There are duplicated (or empty) row IDs!'
        assert(len(V_idx) == len(OrderedSet(V_idx))), 'There are duplicated col IDs!'
        assert(len(V_idx) == V.shape[0]), 'There are duplicated (or empty) col IDs'
 
        # Create SVD
        if VERBOSE:
            sys.stdout.write('Creating SVD() class\n')
        svd = SVD()
        svd._U = DenseMatrix(U, OrderedSet(U_idx), None)
        svd._S = S
        svd._V = DenseMatrix(V, OrderedSet(V_idx), None)
        svd._matrix_similarity = svd._reconstruct_similarity()
        svd._matrix_reconstructed = svd._reconstruct_matrix()

        # If save_model, then use row and col ids from SVDLIBC
        MAX_VECTORS = 2**21
        if len(svd._U) > MAX_VECTORS:
            svd._file_row_ids = file_U_idx
        if len(svd._V) > MAX_VECTORS:
            svd._file_col_ids = file_V_idx
        
        return svd

if __name__ == "__main__":
    import sys
    from recsys.algorithm.factorize import SVD

    datafile = sys.argv[1] #In default matrix format: value \t row \t col \n
    prefix = sys.argv[2]
    matrix = '/tmp/matrix.dat'
    k = int(sys.argv[3])

    svdlibc = SVDLIBC(datafile=datafile, matrix=matrix, prefix=prefix)
    print 'Loading', datafile
    svdlibc.to_sparse_matrix(format={'ids': int})
    svdlibc.compute(k)
    print '\nLoading SVD'
    svd = svdlibc.export()
    print svd
    svd.save_model('/tmp/svd-model', options={'k': k})
