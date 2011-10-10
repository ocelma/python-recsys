Examples
========

You can find all these examples in the *./draft* folder.

Movielens
---------

Simple SVD
~~~~~~~~~~

.. code-block:: python

    import sys

    #To show some messages:
    import recsys.algorithm
    recsys.algorithm.VERBOSE = True

    from recsys.algorithm.factorize import SVD
    from recsys.datamodel.data import Data
    from recsys.evaluation.prediction import RMSE, MAE

    #Dataset
    PERCENT_TRAIN = int(sys.argv[2])
    data = Data()
    print 'Loading dataset %s' % sys.argv[1]
    data.load(sys.argv[1], sep='::', format={'col':0, 'row':1, 'value':2, 'ids':int})
    #Train & Test data
    train, test = data.split_train_test(percent=PERCENT_TRAIN)

    #Create SVD
    K=100
    svd = SVD()
    svd.set_data(train)
    svd.compute(k=K, min_values=5, pre_normalize=None, mean_center=True, post_normalize=True)

    #Evaluation using prediction-based metrics
    rmse = RMSE()
    mae = MAE()
    for rating, item_id, user_id in test.get():
        try:
            pred_rating = svd.predict(item_id, user_id)
            rmse.add(rating, pred_rating)
            mae.add(rating, pred_rating)
        except KeyError:
            continue

    print 'RMSE=%s' % rmse.compute()
    print 'MAE=%s' % mae.compute()

Save it as **movielens.py**, and run it!

.. code-block:: python

    $ python movielens.py tests/data/movielens/ratings.dat 80

    # Here's the output:
    Creating matrix
    Updating matrix: squish to at least 5 values
    Computing svd k=100, min_values=5, pre_normalize=None, mean_center=True, post_normalize=True
    RMSE=0.91919
    MAE=0.717771

Implementing a new algorithm
-----------------------------

Now, here's an example about how to create a new algorithm, by extending *BaseClass* algorithm class.

This Baseline dummy algorithm returns the avg. rating of a user, when predicting the value :math:`\hat{r}_{ui}`, for user :math:`u` and any item :math:`i`

.. code-block:: python

    from numpy import mean
    from operator import itemgetter

    from recsys.algorithm.baseclass import Algorithm

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
                # Vector is a list of tuples: (rating, pos). E.g (3.0, 20)
                self._user_avg_rating[index] = mean(map(itemgetter(0), vector))
            predicted_value = self._user_avg_rating[index]

            if MIN_VALUE:
                predicted_value = max(predicted_value, MIN_VALUE)
            if MAX_VALUE:
                predicted_value = min(predicted_value, MAX_VALUE)
            return predicted_value

Save this example as **baseline.py**

Here's an example using this simple baseline Algorithm class:

.. code-block:: python

    import sys

    #To show some messages:
    import recsys.algorithm
    recsys.algorithm.VERBOSE = True

    from recsys.evaluation.prediction import RMSE, MAE
    from recsys.datamodel.data import Data

    from baseline import Baseline #Import the test class we've just created

    #Dataset
    PERCENT_TRAIN = int(sys.argv[2])
    data = Data()
    print 'Loading dataset %s' % sys.argv[1]
    data.load(sys.argv[1], sep='::', format={'col':0, 'row':1, 'value':2})
    #Train & Test data
    train, test = data.split_train_test(percent=PERCENT_TRAIN)

    baseline = Baseline()
    baseline.set_data(train)
    baseline.compute() # In this case, it does nothing

    # Evaluate
    rmse = RMSE()
    mae = MAE()
    for rating, item_id, user_id in test.get():
        try:
            pred_rating = baseline.predict(item_id, user_id, user_is_row=False)
            rmse.add(rating, pred_rating)
            mae.add(rating, pred_rating)
        except KeyError:
            continue

    print 'RMSE=%s' % rmse.compute()
    print 'MAE=%s' % mae.compute()

Save this example as **test_baseline.py**

And run it:

.. code-block:: python

    $ python test_baseline.py tests/data/movielens/ratings.dat 80

    # Here's the output:
    Loading dataset tests/data/movielens/ratings.dat
    Creating matrix
    RMSE=1.033579
    MAE=0.827535

