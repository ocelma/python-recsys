import os

import warnings
warnings.simplefilter('ignore')

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
MOVIELENS_DATA_PATH = os.path.join(TEST_DATA_PATH, 'movielens')

def skip(x):
    x.__test__ = False
    return x
