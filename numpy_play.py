import numpy as np
from lib2to3.fixer_util import p1

result = np.multiply([1, 2, 3], [1, 2, 3])
print result
print np.sum(result)
print np.dot([1, 2, 3], [1, 2, 3])

# Layered network
h1 = np.dot([1, 2, 3], [1, 1, -5])
h2 = np.dot([1, 2, 3], [3, -4, 2])
print h1, " ", h2
y = np.dot([h1, h2], [2, -1])
print y

# input = [1, 0]
# h1 = np.dot(input, [3, 2])
# h2 = np.dot(input, [-1, 4])
# h3 = np.dot(input, [3, -5])
# print h1, " ", h2, " ", h3
# y = np.dot([h1, h2, h3], [1, 2, -1])
# print y

print "\n AND Perceptron"
input = [1, 1]
h1 = np.dot(input, [.5, .5])
#print "h1=",h1
y = np.dot(h1, 1)
print "neruon =", y
print "linear conversion =", np.dot(input, [0.375, 0.375])


print "\n XOR perceptron"
input = [1, 1, 1]
h1 = np.dot(input, [.5, .5, -1])
y = np.dot(h1, .5)
print y


print "\n\n"
from scipy.spatial import distance
x = [(1,6), (2,4), (3,7), (6,8), (7,1), (8,4)]
y = [7, 8, 16, 44, 50, 68]
p = (4,2)
for i, pair in enumerate(x):
    print "Euclidean distance with {} = {}".format(i, distance.euclidean(pair, p))
    print "Manhattan distance with {} = {}".format(i, distance.cityblock(pair, p))
    print ""

print "Manhattan 1-NN avg: ", np.average([8, 50])
print "Euclidean 3-NN avg: ", np.average([y[1], y[4], y[5]])
print "Manhattan 3-NN avg: ", np.average([y[1], y[4], y[2], y[5]])