from numpy import mean
from scipy.stats import kendalltau, spearmanr, rankdata
from operator import itemgetter

from recsys.evaluation import ROUND_FLOAT
from recsys.evaluation.baseclass import Evaluation

def _compute(f, ground_truth, test):
    elems = len(list(set(map(itemgetter(0), ground_truth)) & set(map(itemgetter(0), test))))
    if len(ground_truth) != elems or len(test) != elems:
        raise ValueError('Ground truth and Test datasets have different elements!')
    ground_truth.sort()
    test.sort()
    ground_truth, test = map(itemgetter(1), ground_truth), map(itemgetter(1), test)
    return round(f(ground_truth, test)[0], ROUND_FLOAT)

#Rank-Based Metrics:
class SpearmanRho(Evaluation):
    def __init__(self, data=None):
        super(SpearmanRho, self).__init__(data)

    def compute(self):
        super(SpearmanRho, self).compute()
        if not len(self._ground_truth) == len(self._test):
            raise ValueError('Ground truth and Test datasets have different sizes!')
        try:
            return _compute(spearmanr, self._ground_truth, self._test)
        except TypeError:
            return round(spearmanr(self._ground_truth, self._test)[0], ROUND_FLOAT)

class KendallTau(Evaluation):
    def __init__(self, data=None):
        super(KendallTau, self).__init__(data)

    def compute(self):
        super(KendallTau, self).compute()
        try:
            return _compute(kendalltau, self._ground_truth, self._test)
        except TypeError:
            return round(kendalltau(self._ground_truth, self._test)[0], ROUND_FLOAT)

class ReciprocalRank(Evaluation):
    def __init__(self):
        super(ReciprocalRank, self).__init__()

    def compute(self, ground_truth=None, query=None):
        if not query:
            query = self._test
        if not ground_truth:
            ground_truth = self._ground_truth
        try:
            rank_query = ground_truth.index(query) + 1
            rr = 1.0 / rank_query
            return rr
        except ValueError:
            return 0.0
        
class MeanReciprocalRank(Evaluation):
    def __init__(self):
        super(MeanReciprocalRank, self).__init__()
        # _rr stores partial ReciprocalRank results:
        self._rr = []

    def load(self, ranked_list, elem):
        if isinstance(elem, list):
            raise ValueError('2nd param must be an element not a list!. For example: load([1,2,3,4], 2)')
        self._ground_truth.append(ranked_list)
        self._test.append(elem)
        #Compute current ReciprocalRank
        rr = ReciprocalRank()
        rr.load(ranked_list, elem)
        self._rr.append(rr.compute())

    def load_test(self, elem):
        raise NotImplementedError("load_test() method not allowed. Use load(ground_truth, query) instead")

    def load_ground_truth(self, gt):
        raise NotImplementedError("load_ground_truth() method not allowed. Use load(ground_truth, query) instead")

    def get_reciprocal_rank_results(self):
        return self._rr

    def compute(self, ground_truth=None, query=None):
        if query and ground_truth:
            self.load(ground_truth, query)
            return self._rr[-1]
        return round(mean(self.get_reciprocal_rank_results()), ROUND_FLOAT)

"""
from recsys.evaluation.decision import PrecisionRecallF1
class PrecisionAtK(Evaluation):
    def __init__(self):
        super(PrecisionAtK, self).__init__()

    def compute(self, ground_truth=None, query=None):
        if not query:
            query = self._test
        if not ground_truth:
            ground_truth = self._ground_truth
        try:
            hit_set = list(set(self._ground_truth) & set(self._test))
            precision = len(hit_set) / float(len(self._test)) #TP/float(TP+FP)
        except ValueError:
            return 0.0
"""

class AveragePrecision(Evaluation):
    def __init__(self):
        super(AveragePrecision, self).__init__()

    def __compute(self):
        super(AveragePrecision, self).compute()
        i = 1
        hits = 0
        p_at_k = [0.0]*len(self._test)
        for item in self._test:
            try:
                hit = self._ground_truth.index(item) + 1
                hits += 1
                p = hits/float(i)
                p_at_k[i-1] = hits/float(i)
            except:
                pass
            i += 1
        return sum(p_at_k)/hits

    def compute(self):
        super(AveragePrecision, self).compute()
        from recsys.evaluation.decision import PrecisionRecallF1

        if not isinstance(self._test, list):
            self._test = [self._test]

        PRF1 = PrecisionRecallF1()
        p_at_k = []
        hits = 0
        for k in range(1, len(self._test)+1):
            test = self._test[:k]
            PRF1.load(self._ground_truth, test)
            if test[k-1] in self._ground_truth:
                p, r, f1 = PRF1.compute()
                hits += 1
            else:
                p = 0.0
            p_at_k.append(p)
        if not hits:
            return 0.0
        return sum(p_at_k)/hits

class MeanAveragePrecision(Evaluation):
    def __init__(self):
        super(MeanAveragePrecision, self).__init__()
        # _ap stores partial AveragePrecision results:
        self._ap = []

    def load(self, ground_truth, test):
        if not isinstance(test, list):
            test = [test]
        self._ground_truth.append(ground_truth)
        self._test.append(test)
        #Compute current AveragePrecision
        ap = AveragePrecision()
        ap.load(ground_truth, test)
        self._ap.append(ap.compute())

    def load_test(self, elem):
        raise NotImplementedError("load_test() method not allowed. Use load(ground_truth, query) instead")

    def load_ground_truth(self, gt):
        raise NotImplementedError("load_ground_truth() method not allowed. Use load(ground_truth, query) instead")

    def get_average_precision_results(self):
        return self._ap

    def compute(self, ground_truth=None, query=None):
        if query and ground_truth:
            self.load(ground_truth, query)
            return self._ap[-1]
        return round(mean(self.get_average_precision_results()), ROUND_FLOAT)

#TODO: how to define utility function g(u,i)? 
# Specially when there's no ratings (R(u,i) = 3, but another value (such as plays)?
#class DCG(Evaluation):
#    pass #TODO
