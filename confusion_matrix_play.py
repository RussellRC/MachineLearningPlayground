# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
from sklearn import datasets

X = pd.read_csv('resources/titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the default settings for train_test_split (or test_size = 0.25 if specified).
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.25, random_state=42)


clf1 = DecisionTreeClassifier()
clf1.fit(features_train, labels_train)
cmDT = confusion_matrix(labels_test, clf1.predict(features_test))
print "Confusion matrix for this Decision Tree:\n", cmDT

clf2 = GaussianNB()
clf2.fit(features_train, labels_train)
cmNB = confusion_matrix(labels_test,clf2.predict(features_test))
print "GaussianNB confusion matrix:\n", cmNB

#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": cmNB,
 "Decision Tree": cmDT
}

'''
Note that the scikit-learn module uses the following transposed form instead:
TN FP
FN TP
'''