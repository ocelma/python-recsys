import sys
import codecs
import pickle
#from random import shuffle
from exceptions import ValueError
from numpy.random import shuffle

from recsys.algorithm import VERBOSE

class Data:
    """
    Handles the relationshops among users and items
    """
    def __init__(self):
        #"""
        #:param data: a list of tuples
        #:type data: list
        #"""
        self._data = list([])

    def __repr__(self):
        s = '%d rows.' % len(self.get())
        if len(self.get()):
            s += '\nE.g: %s' % str(self.get()[0])
        return s

    def __len__(self):
        return len(self.get())

    def __getitem__(self, i):
        if i < len(self._data):
            return self._data[i]
        return None

    def __iter__(self):
        return iter(self.get())

    def set(self, data, extend=False):
        """
        Sets data to the dataset

        :param data: a list of tuples
        :type data: list
        """
        if extend:
            self._data.extend(data)
        else:
            self._data = data

    def get(self):
        """
        :returns: a list of tuples
        """
        return self._data

    def add_tuple(self, tuple):
        """
        :param tuple: a tuple containing <rating, user, item> information (e.g.  <value, row, col>)
        """
        #E.g: tuple = (25, "ocelma", "u2") -> "ocelma has played u2 25 times"
        if not len(tuple) == 3:
            raise ValueError('Tuple format not correct (should be: <value, row_id, col_id>)')
        value, row_id, col_id = tuple
        if not value and value != 0:
            raise ValueError('Value is empty %s' % (tuple,))
        if isinstance(value, basestring):
            raise ValueError('Value %s is a string (must be an int or float) %s' % (value, tuple,))
        if row_id is None or row_id == '':
            raise ValueError('Row id is empty %s' % (tuple,))
        if col_id is None or col_id == '':
            raise ValueError('Col id is empty %s' % (tuple,))
        self._data.append(tuple)

    def split_train_test(self, percent=80, shuffle_data=True):
        """
        Splits the data in two disjunct datasets: train and test

        :param percent: % of training set to be used (test set size = 100-percent)
        :type percent: int
        :param shuffle_data: shuffle dataset?
        :type shuffle_data: Boolean

        :returns: a tuple <Data, Data>
        """
        if shuffle_data:
            shuffle(self._data)
        length = len(self._data)
        train_list = self._data[:int(round(length*percent/100.0))]
        test_list = self._data[-int(round(length*(100-percent)/100.0)):]
        train = Data()
        train.set(train_list)
        test = Data()
        test.set(test_list)

        return train, test

    def load(self, path, force=True, sep='\t', format=None, pickle=False):
        """
        Loads data from a file

        :param path: filename
        :type path: string
        :param force: Cleans already added data
        :type force: Boolean
        :param sep: Separator among the fields of the file content
        :type sep: string
        :param format: Format of the file content. 
            Default format is 'value': 0 (first field), then 'row': 1, and 'col': 2.
            E.g: format={'row':0, 'col':1, 'value':2}. The row is in position 0, 
            then there is the column value, and finally the rating. 
            So, it resembles to a matrix in plain format
        :type format: dict()
        :param pickle: is input file in  pickle format?
        :type pickle: Boolean
        """
        if VERBOSE:
            sys.stdout.write('Loading %s\n' % path)
        if force:
            self._data = list([])
        if pickle:
            self._load_pickle(path)
        else:
            i = 0 
            for line in codecs.open(path, 'r', 'utf8'):
                data = line.strip('\r\n').split(sep)
                value = None
                if not data:
                    raise TypeError('Data is empty or None!')
                if not format:
                    # Default value is 1
                    try:
                        value, row_id, col_id = data
                    except:
                        value = 1
                        row_id, col_id = data
                else:
                    try:
                        # Default value is 1
                        try:
                            value = data[format['value']]
                        except KeyError, ValueError:
                            value = 1
                        try: 
                            row_id = data[format['row']]
                        except KeyError:
                            row_id = data[1]
                        try:
                            col_id = data[format['col']]
                        except KeyError:
                            col_id = data[2]
                        row_id = row_id.strip()
                        col_id = col_id.strip()
                        if format.has_key('ids') and (format['ids'] == int or format['ids'] == 'int'):
                            try:
                                row_id = int(row_id)
                            except:
                                print 'Error (ID is not int) while reading: %s' % data #Just ignore that line
                                continue
                            try:
                                col_id = int(col_id)
                            except:
                                print 'Error (ID is not int) while reading: %s' % data #Just ignore that line
                                continue
                    except IndexError:
                        #raise IndexError('while reading %s' % data)
                        print 'Error while reading: %s' % data #Just ignore that line
                        continue
                # Try to convert ids to int
                try:
                    row_id = int(row_id)
                except: pass
                try:
                    col_id = int(col_id)
                except: pass
                # Add tuple
                try:
                    self.add_tuple((float(value), row_id, col_id))
                except:
                    if VERBOSE:
                        sys.stdout.write('\nError while reading (%s, %s, %s). Skipping this tuple\n' % (value, row_id, col_id))
                    #raise ValueError('%s is not a float, while reading %s' % (value, data))
                i += 1
                if VERBOSE:
                    if i % 100000 == 0:
                        sys.stdout.write('.')
                    if i % 1000000 == 0:
                        sys.stdout.write('|')
                    if i % 10000000 == 0:
                        sys.stdout.write(' (%d M)\n' % int(i/1000000))
            if VERBOSE:
                sys.stdout.write('\n')

    def _load_pickle(self, path):
        """
        Loads data from a pickle file

        :param path: output filename
        :type param: string
        """
        self._data = pickle.load(codecs.open(path))

    def save(self, path, pickle=False):
        """
        Saves data in output file

        :param path: output filename
        :type param: string
        :param pickle: save in pickle format?
        :type pickle: Boolean
        """
        if VERBOSE:
            sys.stdout.write('Saving data to %s\n' % path)
        if pickle:
            self._save_pickle(path)
        else:
            out = codecs.open(path, 'w', 'utf8')
            for value, row_id, col_id in self._data:
                try:
                    value = unicode(value, 'utf8')
                except:
                    if not isinstance(value, unicode):
                        value = str(value)
                try:
                    row_id = unicode(row_id, 'utf8')
                except:
                    if not isinstance(row_id, unicode):
                        row_id = str(row_id)
                try:
                    col_id = unicode(col_id, 'utf8')
                except:
                    if not isinstance(col_id, unicode):
                        col_id = str(col_id)

                s = '\t'.join([value, row_id, col_id])
                out.write(s + '\n')
            out.close()

    def _save_pickle(self, path):
        """
        Saves data in output file, using pickle format

        :param path: output filename
        :type param: string
        """
        pickle.dump(self._data, open(path, "w"))
