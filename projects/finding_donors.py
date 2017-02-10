# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

# Import supplementary visualization code visuals.py
import visuals as vs


from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.utils import resample 

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    if (learner == None):
        results['train_time'] = 0
        results['pred_time'] = 0
        results['acc_train'] = 0
        results['acc_test'] = 0
        results['f_train'] = 0
        results['f_test'] = 0
        return results
    
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[0:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_test[0:300], predictions_train)
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_test[0:300], predictions_train, 0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)
       
    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # Return the results
    return results


def fbeta_scorer(y_true, y_predict):
    score = fbeta_score(y_true, y_predict, .5)
    return score



'''
**********   MAIN BEGIN   **********
'''


# Load the Census dataset
data = pd.read_csv("../resources/census.csv")

# Success - Display the first record
print "\n**** Display 1st record *****"
print data.head(n=1)
print "*****\n"

# TODO: Total number of records
n_records = data.shape[0]
 
# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = data[data['income'] == ">50K"].shape[0]
 
# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = data[data['income'] == "<=50K"].shape[0]
 
# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = float(n_greater_50k*100) / float(n_records)
 
# Print the results
print "\n***** DATA EXPLORATION *****"
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)
print "*****\n"

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

groups = {}
for feature in features_raw.columns:
    groups[feature] = features_raw.groupby([feature]).size()
    print groups[feature]
    print "*****"


# Visualize skewed continuous features of original data
#vs.distribution(data)

skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
#vs.distribution(features_raw, transformed = True)


from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
print "\n**** Display 1st record after log on skewed data and scaling on all numeric features *****"
print features_raw.head(n = 1)
print "*****\n"

# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)


# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.apply(lambda x: 0 if x == "<=50K" else 1)

# Print the number of features after one-hot encoding
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
print "\n***** Encoded Features *****"
print encoded
print "*****\n"



# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])




# TODO: Calculate accuracy
''' proportion of items classified or labeled correctly '''
''' model that always predicted an individual made more than $50,000 '''
accuracy = float(n_greater_50k)/float(n_records)

# TODO: Calculate F-score using the formula above for beta = 0.5
# True Positive / (True Positive + False Positive)
precision = float(n_greater_50k) / float(n_records)
# True Positive / (True Positive + False Negative)  
recall = float(n_greater_50k) / float(n_greater_50k)
#print "precison: {}, recall: {}".format(precision, recall)
beta = .5
fscore = np.square(1 + beta) * (precision * recall) / ((beta * beta * precision) + recall)

# Print the results 
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)



# TODO: Import the three supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

# TODO: Initialize the three models
clf_A = DecisionTreeClassifier(random_state=42)
clf_B = AdaBoostClassifier(random_state=42)
clf_C = None  #This is supposed to be SVC but takes too much time...

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(len(X_train) * .01)
samples_10 = int(len(X_train) * .1)
samples_100 = len(X_train)

# Collect results on the learners
print "\n\n"
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)
        print "*** Results for {}, {} samples:".format(clf_name, samples)
        print results[clf_name][i]
        print "\n"
 
# Run metrics visualization for the three supervised learning models chosen
''' EVALUATION OF 3 SELECTED CLASSIFIERS '''
#vs.evaluate(results, accuracy, fscore)



''' ***** GRID SEARCH ***** '''

# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

# TODO: Initialize the classifier
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=42)
# 
# # TODO: Create the parameters list you wish to tune
# parameters = {'base_estimator__criterion' : ["gini", "entropy"],
#               'base_estimator__splitter' :   ["best"],
#               'base_estimator__min_samples_split' : [5, 50],
#               'n_estimators' : [50, 100, 250], 
#               'learning_rate' : [0.5, 1.0]}
# 
# # TODO: Make an fbeta_score scoring object
# scorer = make_scorer(fbeta_scorer)
# 
# # TODO: Perform grid search on the classifier using 'scorer' as the scoring method
# grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
# 
# # TODO: Fit the grid search object to the training data and find the optimal parameters
# grid_fit = grid_obj.fit(X_train, y_train)
# print "\n\n***** Best params ***** \n", grid_fit.best_params_, "*****\n"
# #print grid_fit.best_score_
# #print grid_fit.grid_scores_
# 
# # Get the estimator
# best_clf = grid_fit.best_estimator_

''' ***** // GRID SEARCH END ***** '''

''' Best parameters found after GridSearch above '''
parameters = {'n_estimators': 100, 
              'learning_rate': 0.5, 
              'base_estimator__criterion': 'entropy', 
              'base_estimator__min_samples_split': 50, 
              'base_estimator__splitter': 'best'}

best_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=42)
best_clf.set_params(**parameters)

start = time()
best_clf.fit(X_train, y_train)
end = time()
train_time_best = end - start


# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)

start = time()
best_predictions = best_clf.predict(X_test)
end = time()
predict_time_best = end - start


# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))




# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import AdaBoostClassifier

# TODO: Train the supervised model on the training set 
model = AdaBoostClassifier(random_state=42)
model.fit(X_train, y_train)

# TODO: Extract the feature importances
importances = model.feature_importances_

# Plot
#vs.feature_plot(importances, X_train, y_train)



# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]


# Train on the "best" model found from grid search earlier
clf = clone(best_clf)
start = time()
clf.fit(X_train_reduced, y_train)
end = time()
train_time_reduced = end - start

# Make new predictions
start = time()
reduced_predictions = clf.predict(X_test_reduced)
end = time()
predict_time_reduced = end - start


# Report scores from the final model using both versions of data
print "Final Model trained on full data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))
print "Train time: {:.4f}".format(train_time_best)
print "Prediction time: {:.4f}".format(predict_time_best)

print "\nFinal Model trained on reduced data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))
print "Train time: {:.4f}".format(train_time_reduced)
print "Prediction time: {:.4f}".format(predict_time_reduced)


''' Uncomment for visuals '''
plt.show()