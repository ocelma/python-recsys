from recsys.evaluation.baseclass import Evaluation
from recsys.evaluation import ROUND_FLOAT

# Decision-Based Metrics. Evaluating Top-N recommendations
class PrecisionRecallF1(Evaluation):
    def __init__(self):
        super(PrecisionRecallF1, self).__init__()

    def add_predicted_value(self, rating_pred): # Synonym of self.add_test
        self.add_test(rating_pred)

    def compute(self):
        super(PrecisionRecallF1, self).compute()
        """
        precision, recall, f1 = (0.0, 0.0, 0.0)
        TP, FP, TN, FN = (0, 0, 0, 0)
        ground_truth = list(self._ground_truth)
        for item in self._test:
            if item in ground_truth:
                TP += 1
                ground_truth.pop(ground_truth.index(item))
            else:
                FP += 1
        FN = len(ground_truth)
        """
        hit_set = list(set(self._ground_truth) & set(self._test))
        precision = len(hit_set) / float(len(self._test)) #TP/float(TP+FP)
        recall = len(hit_set) / float(len(self._ground_truth)) #TP/float(TP+FN)
        if precision == 0.0 and recall == 0.0:
            return (0.0, 0.0, 0.0)
        f1 = 2 * ((precision*recall)/(precision+recall))
        return (round(precision, ROUND_FLOAT), round(recall, ROUND_FLOAT), round(f1, ROUND_FLOAT))

