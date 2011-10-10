from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data

data = [(4.0, 'user1', 'item1'),
 (2.0, 'user1', 'item3'),
 (1.0, 'user2', 'item1'),
 (5.0, 'user2', 'item4')]

d = Data()
d.set(data)
svd = SVD()
svd.set_data(d)
m = svd.get_matrix()
svd.compute(k=2)
print svd.similar('user1')
print svd.predict('user1', 'item1')
