import sys
from numpy import nan, mean

#To show some messages:
import recsys.algorithm
recsys.algorithm.VERBOSE = True

from recsys.algorithm.factorize import SVD, SVDNeighbourhood
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE

# Create SVD
K=100
svd = SVD()
svd_neig = SVDNeighbourhood()

#Dataset
PERCENT_TRAIN = int(sys.argv[2])
data = Data()
data.load(sys.argv[1], sep='::', format={'col':0, 'row':1, 'value':2, 'ids': int})

rmse_svd_all = []
mae_svd_all = []
rmse_svd_neig_all = []
mae_svd_neig_all = []

RUNS = [1,2,3,4,5,6,7,8,9,10]
for run in RUNS:
	print 'RUN(%d)' % run
	#Train & Test data
	train, test = data.split_train_test(percent=PERCENT_TRAIN)

	svd.set_data(train)
	svd_neig.set_data(train)

	#Compute SVD
	svd.compute(k=K, min_values=None, pre_normalize=None, mean_center=True, post_normalize=True)
	svd_neig.compute(k=K, min_values=None, pre_normalize=None, mean_center=True, post_normalize=True)

	# Evaluate
	rmse_svd = RMSE()
	mae_svd = MAE()
	rmse_svd_neig = RMSE()
	mae_svd_neig = MAE()

	i = 1
	total = len(test.get())
	print 'Total Test ratings: %s' % total
	for rating, item_id, user_id in test:
	    try:
		    pred_rating_svd = svd.predict(item_id, user_id)
		    rmse_svd.add(rating, pred_rating_svd)
		    mae_svd.add(rating, pred_rating_svd)

		    pred_rating_svd_neig = svd_neig.predict(item_id, user_id) #Koren & co.
		    if pred_rating_svd_neig is not nan:
		        rmse_svd_neig.add(rating, pred_rating_svd_neig)
		        mae_svd_neig.add(rating, pred_rating_svd_neig)

		    print "\rProcessed test rating %d" % i,                                        
		    sys.stdout.flush()

		    i += 1
	    except KeyError:
    		continue

	    rmse_svd_all.append(rmse_svd.compute())
	    mae_svd_all.append(mae_svd.compute())
	    rmse_svd_neig_all.append(rmse_svd_neig.compute())
	    mae_svd_neig_all.append(mae_svd_neig.compute())
print
print 'RMSE (SVD) = %s | STD = %s' % (mean(rmse_svd_all), std(rmse_svd_all))
print 'MAE  (SVD) = %s | STD = %s' % (mean(mae_svd_all), std(mae_svd_all))
print 'RMSE (SVD Neig.) = %s | STD = %s' % (mean(rmse_svd_neig_all), std(rmse_svd_neig_all))
print 'MAE  (SVD Neig.) = %s | STD = %s' % (mean(mae_svd_neig_all), std(mae_svd_neig_all))
