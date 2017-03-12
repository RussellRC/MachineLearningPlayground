"""Softmax."""

import numpy as np
import time

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=0)
    sigma = exp_x / sum_exp_x
    #print np.sum(sigma)
    return sigma
        
scores = np.array([3.0, 1.0, 0.2])
print(softmax(scores))

# Probabilities get closer to either 0 or 1
print(softmax(scores * 10))  

# Probabilities get closer to Normal Distribution, 
# since all the scores decrease in magnitude, 
# the resulting softmax probabilities will be closer to each other.
print(softmax(scores / 10))


# print(softmax([1.0, 2.0, 3.0]))
# scores = np.array([[1, 2, 3, 6],
#                    [2, 4, 5, 6],
#                    [3, 8, 7, 6]])
# print(softmax(scores))
# 
# 
# 
# #Plot softmax curves
# import matplotlib.pyplot as plt
# x = np.arange(-2.0, 6.0, 0.1)
# scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
#  
# plt.plot(x, softmax(scores).T, linewidth=2)
# plt.show()


n = 1000000000
n = n + 0.000001
n = n - 1000000000
print "{}".format(n)


a = np.array([True, True, False, False])
b = np.array([True, False, True, False])
c = (a | b)
print c

t0 = time.time()
t1 = time.time()
print "train time for size '{}': {:.4f}".format(50, t1-t0)