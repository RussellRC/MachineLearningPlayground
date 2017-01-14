from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

import numpy as np


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))


sw = stopwords.words("english")
print sw[0]
print len(sw)