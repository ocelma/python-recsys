import codecs
import os
from operator import itemgetter
import csv
from numpy import array
from divisi2.ordered_set import OrderedSet
from csc.divisi2.dense import DenseMatrix
from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data

# Path to 'svd' executable [ http://tedlab.mit.edu/~dr/SVDLIBC/ ]
PATH_SVDLIBC = '/usr/local/bin/'

class SVDLIBC(object):
    def __init__(self, datafile, matrix='matrix.dat', prefix='svd'):
        self._data_file = datafile
        self._matrix_file = matrix
        self._svd_prefix = prefix

    def compute(self, k=100):
        os.spawnv(os.P_WAIT, PATH_SVDLIBC + 'svd', ['-r st', '-d%d' % k, '-o%s' % self._svd_prefix, self._matrix_file])

    def to_sparse_matrix(self, sep='\t', format={'ids': int}):
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
            if isinstance(row_id, int):
                row_id = str(row_id)
            f_row_ids.write(row_id + '\n')
        f_row_ids.close()
        cols = cols.items()
        cols.sort(key=itemgetter(1))
        for col_id, _ in cols:
            if isinstance(col_id, int):
                col_id = str(col_id)
            #f_col_ids.write(unicode(col_id, 'utf8') + '\n')
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
        try:
            U_idx = [ int(idx.strip()) for idx in open(file_U_idx)]
        except:
            U_idx = [ idx.strip() for idx in open(file_U_idx)]
        try:
            V_idx = [ int(idx.strip()) for idx in open(file_V_idx)]
        except:
            V_idx = [ idx.strip() for idx in open(file_V_idx)]
        
        #Check no duplicated IDs!!!
        assert(len(U_idx) == len(OrderedSet(U_idx)))
        assert(len(V_idx) == len(OrderedSet(V_idx)))
        
        # Create SVD
        svd = SVD()
        svd._U = DenseMatrix(U, OrderedSet(U_idx), None)
        svd._S = S
        svd._V = DenseMatrix(V, OrderedSet(V_idx), None)
        svd._matrix_similarity = svd._reconstruct_similarity()
        svd._matrix_reconstructed = svd._reconstruct_matrix()
        
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
    svdlibc.to_sparse_matrix()
    svdlibc.compute(k)
    print '\nLoading SVD'
    svd = svdlibc.export()
    print svd
    svd.save_model('/tmp/MODEL')
