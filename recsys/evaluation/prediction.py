from math import sqrt
from scipy.stats import pearsonr

from recsys.evaluation import ROUND_FLOAT
from recsys.evaluation.baseclass import Evaluation

#Predictive-Based Metrics
class MAE(Evaluation):
    """
    Mean Absolute Error

    :param data: a tuple containing the Ground Truth data, and the Test data
    :type data: <list, list>
    """
    def __init__(self, data=None):
        super(MAE, self).__init__(data)

    def compute(self, r=None, r_pred=None):
        if r and r_pred:
            return round(abs(r - r_pred), ROUND_FLOAT)

        if not len(self._ground_truth) == len(self._test):
            raise ValueError('Ground truth and Test datasets have different sizes!')        

        #Compute for the whole test set
        super(MAE, self).compute()
        sum = 0.0 
        for i in range(0, len(self._ground_truth)):
            r = self._ground_truth[i]
            r_pred = self._test[i]
            sum += abs(r - r_pred)
        return round(abs(float(sum/len(self._test))), ROUND_FLOAT)

class RMSE(Evaluation):
    """
    Root Mean Square Error

    :param data: a tuple containing the Ground Truth data, and the Test data
    :type data: <list, list>
    """
    def __init__(self, data=None):
        super(RMSE, self).__init__(data)

    def compute(self, r=None, r_pred=None):
        if r and r_pred:
            return round(sqrt(abs((r - r_pred)*(r - r_pred))), ROUND_FLOAT)

        if not len(self._ground_truth) == len(self._test):
            raise ValueError('Ground truth and Test datasets have different sizes!')        

        #Compute for the whole test set
        super(RMSE, self).compute()
        sum = 0.0 
        for i in range(0, len(self._ground_truth)):
            r = self._ground_truth[i]
            r_pred = self._test[i]
            sum += abs((r - r_pred)*(r - r_pred))
        return round(sqrt(abs(float(sum/len(self._test)))), ROUND_FLOAT)

#Correlation-Based Metrics
class Pearson(Evaluation):
    """
    Pearson correlation

    :param data: a tuple containing the Ground Truth data, and the Test data
    :type data: <list, list>
    """
    def __init__(self, data=None):
        super(Pearson, self).__init__(data)

    def compute(self):
        super(Pearson, self).compute()
        return round(pearsonr(self._ground_truth, self._test)[0], ROUND_FLOAT)
