import numpy as np
import pandas as pd

import visuals as vs
import matplotlib.pyplot as pl
from sklearn.decomposition.tests.test_nmf import random_state

pd.set_option('display.width', 800)

data = pd.read_csv('../resources/customers.csv')

SHOW_VISUALS = False


print '''
/* 
 * ************************************** 
 * ********** DATA EXPLORATION **********
 * **************************************
 */
'''
data.drop(['Region', 'Channel'], axis = 1, inplace = True)
print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)

print data.describe(),"\n"

# print data[data['Fresh'] == 3]
print data[data['Fresh'] == 112151]
print "\n"
# print data[data['Detergents_Paper'] == 3]
print data[data['Detergents_Paper'] == 40827]
print "\n"
# print data[data['Delicatessen'] == 3]
print data[data['Delicatessen'] == 47943]
print "\n"


# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [181, 85, 183]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
print samples
print "\n\n"

print '''
/* 
 * ***** Feature Relevance *****
 */
'''

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
for column in data.columns:
    new_data = data.drop([column], axis = 1)
    Y = data[column]
    #print "new_data.columns ", new_data.columns
    #print "Y.shape ", Y.shape
    
    # TODO: Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data, Y, test_size=0.25, random_state=42)
    
    # TODO: Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(X_train, y_train)
    
    # TODO: Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)
    print "score on guessing '{}': {}".format(column, score)
    
if SHOW_VISUALS:
    axes = pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


print "\n\n"
print '''
/* 
 * **************************************** 
 * ********** DATA PREPROCESSING **********
 * ****************************************
 */
'''

print '''
/* 
 * ***** Feature scaling *****
 */
'''

# TODO: Scale the data using the natural logarithm
log_data = data.apply(np.log)

# TODO: Scale the sample data using the natural logarithm
log_samples = samples.apply(np.log)

if SHOW_VISUALS:
    # Produce a scatter matrix for each pair of newly-transformed features
    pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

print "Log samples:"
print log_samples
print "\n\n"


print '''
/* 
 * ***** Outlier Detection *****
 */
'''

# For each feature find the data points with extreme high or low values
outliers_count = {}
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1) * 1.5
    
    # Display the outliers
    print "\n"
    print "Data points considered outliers for the feature '{}':".format(feature)
    feature_outliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    print feature_outliers
    
    for customer_index in feature_outliers.index:
        outlier_features = [feature];
        if customer_index in outliers_count:
            outlier_features = outliers_count[customer_index]
            outlier_features.append(feature)
        outliers_count[customer_index] = outlier_features

print "\n"
print "Total outliers count"
print sorted(outliers_count.items(), key=lambda x: len(x[1]), reverse=True)

# OPTIONAL: Select the indices for data points you wish to remove
outliers = []
for k,v in outliers_count.items():
    if len(v) > 1:
        outliers.append(k)
print outliers
outliersPD = pd.DataFrame(data.loc[outliers], columns = data.keys())
print outliersPD 


# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


print '''
/* 
 * ***** PCA Analysis *****
 */
'''

from sklearn.decomposition import PCA 

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(len(good_data.keys()))
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

if SHOW_VISUALS:
    # Generate PCA results plot
    pca_results = vs.pca_results(good_data, pca)


print "PCA with 6 components Explained Variance Ratio"
print pca.explained_variance_ratio_
print "Total variance of PCA 1 and 2: ", pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
print "Total variance of PCA 1 to 4: ", pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]  + pca.explained_variance_ratio_[2] + pca.explained_variance_ratio_[3]


print '''
/* 
 * ***** PCA Application *****
 */
'''

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)
pca.fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

print "3 Scaled Samples with PCA=2..."
print pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2'])

if SHOW_VISUALS:
    ax = vs.biplot(good_data, reduced_data, pca)


print '''
/* 
 * ************************************ 
 * ********** IMPLEMENTATION **********
 * ************************************
 */
'''
print '''
/* 
 * ***** Creating Clusters *****
 */
'''
from sklearn.mixture import GMM
from sklearn.metrics import silhouette_score

#mixtures = np.arange(2, 6)
mixtures = np.arange(2, 3)
    
for n_components in mixtures:
    # TODO: Apply your clustering algorithm of choice to the reduced data 
    clusterer = GMM(n_components=n_components, random_state=42)
    clusterer.fit(reduced_data)
    
    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)
    
    # TODO: Find the cluster centers
    centers = clusterer.means_
    print "centers: "
    print centers
    
    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)
    
    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)
    print "Silhouette score with {} clusters: {}".format(n_components, np.round(score, 4))
    if SHOW_VISUALS:
        vs.cluster_results(reduced_data, preds, centers, pca_samples)


print '''
/* 
 * ***** Data recovery *****
 */
'''
# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
print true_centers


vs.channel_results(reduced_data, outliers, pca_samples)


#if SHOW_VISUALS:
pl.show()