from operator import itemgetter
from numpy import nan

class Evaluation(object):
    """
    Base class for Evaluation

    It has the basic methods to load ground truth and test data.
    Any other Evaluation class derives from this base class.

    :param data: A list of tuples, containing the real and the predicted value. E.g: [(3, 2.3), (1, 0.9), (5, 4.9), (2, 0.9), (3, 1.5)]
    :type data: list
    """
    def __init__(self, data=None):
        #data is a list of tuples. E.g: [(3, 2.3), (1, 0.9), (5, 4.9), (2, 0.9), (3, 1.5)]
        if data:
            self._ground_truth, self._test = map(itemgetter(0), data), map(itemgetter(1), data)
        else:
            self._ground_truth = []
            self._test = []

    def __repr__(self):
        gt = str(self._ground_truth)
        test = str(self._test)
        return 'GT  : %s\nTest: %s' % (gt, test)
        #return str('\n'.join((str(self._ground_truth), str(self._test))))

    def load_test(self, test):
        """
        Loads a test dataset

        :param test: a list of predicted values. E.g: [2.3, 0.9, 4.9, 0.9, 1.5] 
        :type test: list
        """
        if isinstance(test, list):
            self._test = list(test)
        else:
            self._test = test

    def get_test(self):
        """
        :returns: the test dataset (a list)
        """
        return self._test

    def load_ground_truth(self, ground_truth):
        """
        Loads a ground truth dataset

        :param ground_truth: a list of real values (aka ground truth). E.g: [3.0, 1.0, 5.0, 2.0, 3.0]
        :type ground_truth: list
        """
        if isinstance(ground_truth, list):
            self._ground_truth = list(ground_truth)
        else:
            self._ground_truth = ground_truth

    def get_ground_truth(self):
        """
        :returns: the ground truth list
        """
        return self._ground_truth

    def load(self, ground_truth, test):
        """
        Loads both the ground truth and the test lists. The two lists must have the same length.

        :param ground_truth: a list of real values (aka ground truth). E.g: [3.0, 1.0, 5.0, 2.0, 3.0]
        :type ground_truth: list
        :param test: a list of predicted values. E.g: [2.3, 0.9, 4.9, 0.9, 1.5] 
        :type test: list
        """
        self.load_ground_truth(ground_truth)
        self.load_test(test)

    def add(self, rating, rating_pred):
        """
        Adds a tuple <real rating, pred. rating>

        :param rating: a real rating value (the ground truth)
        :param rating_pred: the predicted rating
        """
        if rating is not nan and rating_pred is not nan:
            self._ground_truth.append(rating)
            self._test.append(rating_pred)

    def add_test(self, rating_pred):
        """
        Adds a predicted rating to the current test list

        :param rating_pred: the predicted rating
        """
        if rating_pred is not nan:
            self._test.append(rating_pred)

    def compute(self):
        """
        Computes the evaluation using the loaded ground truth and test lists
        """
        if len(self._ground_truth) == 0:
            raise ValueError('Ground Truth dataset is empty!')
        if len(self._test) == 0:
            raise ValueError('Test dataset is empty!')

