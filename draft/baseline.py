from numpy import mean
from operator import itemgetter

from recsys.algorithm.baseclass import Algorithm
from recsys.algorithm import VERBOSE

class Baseline(Algorithm):
    def __init__(self):
        #Call parent constructor
        super(Baseline, self).__init__()

        # 'Cache' for user avg. rating
        self._user_avg_rating = dict()

    def predict(self, i, j, MIN_VALUE=None, MAX_VALUE=None, user_is_row=True):
        index = i
        if not user_is_row:
            index = j

        if not self._user_avg_rating.has_key(index):
            if user_is_row:
                vector = self.get_matrix().get_row(index).entries()
            else:
                vector = self.get_matrix().get_col(index).entries()
            # Vector is a list of tuples: (rating, index). E.g (3.0, 20)
            self._user_avg_rating[index] = mean(map(itemgetter(0), vector))
        predicted_value = self._user_avg_rating[index]

        if MIN_VALUE:
            predicted_value = max(predicted_value, MIN_VALUE)
        if MAX_VALUE:
            predicted_value = min(predicted_value, MAX_VALUE)
        return predicted_value
