from nltk.corpus import stopwords
from numpy import dtype
import numpy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.utils import resample


import matplotlib.pyplot as plt
from sklearn.decomposition.tests.test_nmf import random_state


# def do_stuff():
#     print "hi"
#     
# l = [1, 0, 1, 1, 0]
# ones = filter(lambda e: e is 1, l)
# print ones
# 
# do_stuff()
weights = numpy.array([[10, 50], [50, 25], [100, 0]], dtype=float)
print weights
print "#####"
scaler1 = MinMaxScaler()
scaler1.fit(weights)
minmax_weights = scaler1.transform(weights)
print minmax_weights
print "#####"

scaler2 = StandardScaler()
scaler2.fit(weights)
std_weights = scaler2.transform(weights)
print std_weights
print "#####"

scaler3 = RobustScaler()
scaler3.fit(weights)
rob_weights = scaler3.transform(weights)
print rob_weights
print "#####"


sw = stopwords.words("english")

data = numpy.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(data)
print transformed_data

# plt.figure()
# for i, j in zip(data, transformed_data):
#     plt.scatter(i[0], i[1], color="b")
#     plt.scatter(j[0], j[1], color="r")
# plt.show()



print numpy.arange(5,11)

x = [1, 2, 3, 4, 5]
y = [-1, -2, -3, -4, -5]
x1, y1 = resample(x, y, n_samples=3, random_state=42)
print x
print y
print x1
print y1