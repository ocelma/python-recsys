from nose import with_setup
from nose.tools import assert_equal, assert_not_equal, assert_raises, assert_true
from numpy import nan, array

from operator import itemgetter

from recsys.evaluation.prediction import MAE, RMSE, Pearson
from recsys.evaluation.decision import PrecisionRecallF1
from recsys.evaluation.ranking import KendallTau, SpearmanRho, MeanReciprocalRank, ReciprocalRank, AveragePrecision, MeanAveragePrecision

class Test(object):
    def __init__(self):
        self.DATA_PRED = [(3, 2.3), (1, 0.9), (5, 4.9), (2, 0.9), (3, 1.5)]
        self.GT_DATA = map(itemgetter(0), self.DATA_PRED)
        self.TEST_DATA = map(itemgetter(1), self.DATA_PRED)

        self.TEST_DECISION = ['classical', 'invented', 'baroque', 'instrumental']
        self.GT_DECISION = ['classical', 'instrumental', 'piano', 'baroque']

        self.TEST_RANKING = [('classical', 25.0), ('piano', 75.0), ('baroque', 50.0), ('instrumental', 25.0)]
        self.GT_RANKING = [('classical', 50.0), ('piano', 100.0), ('baroque', 25.0), ('instrumental', 25.0)]


class TestPrediction(Test):
    def __init__(self):
        super(TestPrediction, self).__init__()
        # Prediction-based metrics: MAE, RMSE, Pearson
        self.mae = MAE(self.DATA_PRED)
        self.rmse = RMSE(self.DATA_PRED)

        self.R = 3        # Real Rating (ground truth)
        self.R_PRED = 2.1 # Predicted Rating

    # test_PRED MAE
    def test_PRED_MAE_compute_one(self):
        assert_equal(self.mae.compute(self.R, self.R_PRED), 0.9)

    def test_PRED_MAE_compute_one_empty_datasets(self):
        mae = MAE()
        assert_equal(mae.compute(self.R, self.R_PRED), 0.9)

    def test_PRED_MAE_compute_all(self):
        assert_equal(self.mae.compute(), 0.7)

    def test_PRED_MAE_nan(self):
        mae = MAE()
        mae.add(2.0, nan)
        assert_equal(mae.get_test(), [])
        assert_equal(mae.get_ground_truth(), [])

    def test_PRED_MAE_load(self):
        mae = MAE()
        mae.load(self.GT_DATA, self.TEST_DATA)
        assert_equal(mae.compute(), 0.7)

    def test_PRED_MAE_load_test(self):
        mae = MAE()
        mae.load_test(self.TEST_DATA)
        assert_equal(len(mae.get_test()), len(self.TEST_DATA))
        assert_equal(len(mae.get_ground_truth()), 0)
        assert_raises(ValueError, mae.compute) #Raise: GT is empty!

    def test_PRED_MAE_load_test_and_ground_truth(self):
        mae = MAE()
        mae.load_test(self.TEST_DATA)
        mae.load_ground_truth(self.GT_DATA)
        assert_equal(mae.compute(), 0.7)

    def test_PRED_MAE_add_entry(self):
        self.mae.add(1, 4) #1: GT rating, 4: Predicted rating
        assert_equal(len(self.mae.get_test()), len(self.DATA_PRED)+1)
        assert_equal(self.mae.compute(), 1.083333)

    def test_PRED_MAE_different_list_sizes(self):
        mae = MAE()
        GT = [3, 1, 5, 2]
        # GT list has one element less than self.TEST_DATA
        mae.load(GT, self.TEST_DATA)
        assert_raises(ValueError, mae.compute)

    # test_PRED RMSE
    def test_PRED_RMSE_compute_one(self):
        #Even though rmse has data, we only compute these two param values
        assert_equal(self.rmse.compute(self.R, self.R_PRED), 0.9)

    def test_PRED_RMSE_compute_one_empty_datasets(self):
        rmse = RMSE()
        assert_equal(rmse.compute(self.R, self.R_PRED), 0.9)

    def test_PRED_RMSE_compute_all(self):
        assert_equal(self.rmse.compute(), 0.891067)

    def test_PRED_RMSE_load_test(self):
        rmse = RMSE()
        self.TEST_DATA = [2.3, 0.9, 4.9, 0.9, 1.5]
        rmse.load_test(self.TEST_DATA)
        assert_equal(len(rmse.get_test()), len(self.TEST_DATA))

    def test_PRED_RMSE_add_entry(self):
        self.rmse.add(1,4)
        assert_equal(len(self.rmse.get_test()), len(self.DATA_PRED)+1)
        assert_equal(self.rmse.compute(), 1.470261)

    def test_PRED_RMSE_different_list_sizes(self):
        rmse = RMSE()
        GT = [3, 1, 5, 2]
        # GT list has one element less than self.TEST_DATA
        rmse.load(GT, self.TEST_DATA)
        assert_raises(ValueError, rmse.compute)

    def test_PRED_RMSE_numpy_array(self):
        rmse = RMSE()
        rmse.load(array(self.GT_DATA), array(self.TEST_DATA))
        assert(rmse.compute(), 0.891067)

# TEST_DECISION P/R/F1
class TestDecision(Test):
    def __init__(self):
        super(TestDecision, self).__init__()
        # Decision-based metrics: PrecisionRecallF1
        self.decision = PrecisionRecallF1()
        self.decision.load(self.GT_DECISION, self.TEST_DECISION)

    def test_decision_PRF1_compute_all(self):
        assert_equal(self.decision.compute(), (0.75, 0.75, 0.75)) #P, R, F1
        assert_equal(self.decision.compute(), (0.75, 0.75, 0.75))

    def test_decision_PRF1_empty(self):
        decision = PrecisionRecallF1()
        assert_raises(ValueError, decision.compute)

    def test_decision_PRF1_load_test(self):
        decision = PrecisionRecallF1()
        decision.load_test(self.TEST_DECISION)
        assert_equal(len(decision.get_test()), len(self.TEST_DECISION))

    def test_decision_PRF1_load_ground_truth(self):
        decision = PrecisionRecallF1()
        decision.load_ground_truth(self.GT_DECISION)
        assert_equal(len(decision.get_ground_truth()), len(self.GT_DECISION))

    def test_decision_PRF1_load_test_and_ground_truth(self):
        decision = PrecisionRecallF1()
        decision.load_test(self.TEST_DECISION)
        assert_equal(len(decision.get_test()), len(self.TEST_DECISION))
        decision.load_ground_truth(self.GT_DECISION)
        assert_equal(len(decision.get_ground_truth()), len(self.GT_DECISION))
        P, R, F1 = decision.compute()
        assert_equal(P, 0.75)
        assert_equal(R, 0.75)
        assert_equal(F1, 0.75)

    def test_decision_PRF1_add_entry(self):
        self.decision.add_predicted_value('guitar') #add_predicted_entry == add_test_entry
        assert_equal(len(self.decision.get_test()), len(self.TEST_DECISION)+1)
        assert_equal(len(self.decision.get_ground_truth()), len(self.GT_DECISION))
        P, R, F1 = self.decision.compute()
        assert_equal(P, 0.6)
        assert_equal(R, 0.75)
        assert_equal(F1, 0.666667)


# TEST_CORR Pearson
class TestCorrelation(Test):
    def __init__(self):
        super(TestCorrelation, self).__init__()
        self.pearson = Pearson(self.DATA_PRED)

    def test_CORR_Pearson_compute_all(self):
        assert_equal(self.pearson.compute(), 0.930024)

    def test_CORR_Pearson_load_test(self):
        pearson = Pearson()
        pearson.load_test(self.TEST_DATA)
        assert_equal(len(pearson.get_test()), len(self.TEST_DATA))

    def test_CORR_Pearson_load_ground_truth(self):
        pearson = Pearson()
        pearson.load_ground_truth(self.GT_DATA)
        assert_equal(len(pearson.get_ground_truth()), len(self.GT_DATA))

    def test_CORR_Pearson_add_entry(self):
        self.pearson.add(1, 4) #1: Real rating, 4: Predicted rating
        assert_equal(len(self.pearson.get_test()), len(self.DATA_PRED)+1)
        assert_equal(len(self.pearson.get_ground_truth()), len(self.DATA_PRED)+1)
        assert_equal(self.pearson.compute(), 0.498172)


class TestRanking(Test):
    def __init__(self):
        super(TestRanking, self).__init__()
        # Rank-based metrics:  KendallTau, SpearmanRho, MeanReciprocalRank, ReciprocalRank
        self.kendall = KendallTau()
        self.kendall.load(self.GT_RANKING, self.TEST_RANKING)
        self.spearman = SpearmanRho()
        self.spearman.load(self.GT_RANKING, self.TEST_RANKING)
        self.mrr = MeanReciprocalRank()

        for elem in self.TEST_DECISION:
            self.mrr.load(self.GT_DECISION, elem)

    # TEST_CORR Spearman
    def test_RANK_Spearman_compute_all(self):
        assert_equal(self.spearman.compute(), 0.5) #0.55 ?

    #def test_RANK_Spearman_compute_tied_ranks():
    #    assert_equal(spearman.compute(tied_ranks=True), 0.5) #In fact, it uses Pearsonr corr. of the ranks

    def test_RANK_Spearman_compute_floats(self):
        spearman = SpearmanRho(self.DATA_PRED)
        assert_equal(spearman.compute(), 0.947368) #0.95 ?

    #def test_RANK_Spearman_compute_floats_tied_ranks():
    #    spearman = SpearmanRho(self.DATA_PRED)
    #    assert_equal(spearman.compute(tied_ranks=True), 0.930024) #In fact, it uses Pearsonr corr. of the ranks

    def test_RANK_Spearman_load_test(self):
        spearman = SpearmanRho()
        spearman.load_test(self.TEST_DATA)
        assert_equal(len(spearman.get_test()), len(self.TEST_DATA))

    def test_RANK_Spearman_load_ground_truth(self):
        spearman = SpearmanRho()
        spearman.load_ground_truth(self.GT_DATA)
        assert_equal(len(spearman.get_ground_truth()), len(self.TEST_DATA))

    def test_RANK_Spearman_add_entry(self):
        self.spearman.add(('guitar', 4), ('guitar', 4)) #add tag 'guitar' at rank-4
        assert_equal(len(self.spearman.get_test()), len(self.TEST_RANKING)+1)
        assert_equal(len(self.spearman.get_ground_truth()), len(self.GT_RANKING)+1)
        assert_equal(self.spearman.compute(), 0.763158) #0.775 ?

    def test_RANK_Spearman_different_list_sizes(self):
        TEST_DATA = ['classical', 'invented', 'baroque']
        GT_DATA = ['classical', 'instrumental', 'piano', 'baroque']
        spearman = SpearmanRho()
        spearman.load_ground_truth(GT_DATA)
        spearman.load_test(TEST_DATA)
        assert_raises(ValueError, spearman.compute) #Raise: GT & TEST list have different sizes

    # TEST_CORR Kendall
    def test_RANK_Kendall_compute_all(self):
        assert_equal(self.kendall.compute(), 0.4)

    def test_RANK_Kendall_compute_floats(self):
        kendall = KendallTau(self.DATA_PRED)
        assert_equal(kendall.compute(), 0.888889)

    def test_RANK_Kendall_load_test(self):
        kendall = KendallTau()
        kendall.load_test(self.TEST_DATA)
        assert_equal(len(kendall.get_test()), len(self.TEST_DATA))

    def test_RANK_Kendall_load_ground_truth(self):
        kendall = KendallTau()
        kendall.load_ground_truth(self.GT_DATA)
        assert_equal(len(kendall.get_ground_truth()), len(self.GT_DATA))

    def test_RANK_Kendall_add_entry(self):
        self.kendall.add(('guitar', 4.0), ('guitar', 4.0)) #add tag 'guitar'
        assert_equal(len(self.kendall.get_test()), len(self.TEST_RANKING)+1)
        assert_equal(len(self.kendall.get_ground_truth()), len(self.GT_RANKING)+1)
        assert_equal(self.kendall.compute(), 0.666667)

    def test_RANK_Kendall_diff_elems(self):
        TEST_DECISION = ['class', 'invented', 'baro', 'instru']
        GT_DECISION = ['classical', 'instrumental', 'piano', 'baroque']
        kendall = KendallTau()
        kendall.load_ground_truth(self.GT_DECISION)
        kendall.load_test(self.TEST_DECISION)
        assert_raises(ValueError, kendall.compute) #Different elements

    # TEST_RANK ReciprocalRank
    def test_RANK_ReciprocalRank_compute(self):
        rr = ReciprocalRank()
        QUERY = 'instrumental'
        assert_equal(rr.compute(self.GT_DECISION, QUERY), 0.5)

    def test_RANK_ReciprocalRank_add_entry(self):
        rr= ReciprocalRank()
        QUERY = 'invented'
        rr.load(self.GT_DECISION, QUERY)
        assert_equal(rr.compute(), 0.0)

    # TEST_RANK MeanReciprocalRank
    # Internally, MeanReciprocalRank uses a list of ReciprocalRank results
    def test_RANK_MeanReciprocalRank_compute_all(self):
        assert_equal(self.mrr.compute(), 0.4375)

    def test_RANK_MeanReciprocalRank_compute_one(self):
        mrr  = MeanReciprocalRank()
        QUERY = 'instrumental'
        assert_equal(mrr.compute(self.GT_DECISION, QUERY), 0.5)

    def test_RANK_MeanReciprocalRank_load(self):
        mrr  = MeanReciprocalRank()
        assert_raises(ValueError, mrr.load, self.GT_DECISION, self.TEST_RANKING)

    def test_RANK_MeanReciprocalRank_load_test(self):
        mrr  = MeanReciprocalRank()
        assert_raises(NotImplementedError, mrr.load_test, self.TEST_RANKING)

    def test_RANK_MeanReciprocalRank_load_ground_truth(self):
        mrr  = MeanReciprocalRank()
        assert_raises(NotImplementedError, mrr.load_ground_truth, self.GT_RANKING)

    def test_RANK_MeanReciprocalRank_add_entry(self):
        mrr  = MeanReciprocalRank()
        QUERY = 'invented'
        mrr.load(self.GT_DECISION, QUERY)
        assert_equal(mrr.compute(), 0.0)

    #mAP tests
    def test_RANK_AveragePrecision(self):
        GT_DECISION = [1, 2, 4]
        TEST_DECISION = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        avgp = AveragePrecision()
        avgp.load(GT_DECISION, TEST_DECISION)
        assert_equal(round(avgp.compute(), 4), 0.9167)

        GT_DECISION = [1, 4, 8]
        avgp = AveragePrecision()
        avgp.load(GT_DECISION, TEST_DECISION)
        assert_equal(round(avgp.compute(), 4), 0.625)

        GT_DECISION = [3, 5, 9, 25, 39, 44, 56, 71, 89, 123]
        TEST_DECISION = [123, 84, 56, 6, 8, 9, 511, 129, 187, 25, 38, 48, 250, 113, 3]
        avgp = AveragePrecision()
        avgp.load(GT_DECISION, TEST_DECISION)
        assert_equal(avgp.compute(), 0.58)

    #mAP tests
    def test_RANK_MeanAveragePrecision(self):
        mavgp = MeanAveragePrecision()
        GT_DECISION = [1, 2, 4]
        TEST_DECISION = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mavgp.load(GT_DECISION, TEST_DECISION)

        GT_DECISION = [1, 4, 8]
        mavgp.load(GT_DECISION, TEST_DECISION)

        GT_DECISION = [3, 5, 9, 25, 39, 44, 56, 71, 89, 123]
        TEST_DECISION = [123, 84, 56, 6, 8, 9, 511, 129, 187, 25, 38, 48, 250, 113, 3]
        mavgp.load(GT_DECISION, TEST_DECISION)

        assert_equal(mavgp.compute(), 0.707222)

