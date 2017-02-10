import numpy
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
             'Netherlands', 'Germany', 'Switzerland', 'Belarus',
             'Austria', 'France', 'Poland', 'China', 'Korea', 
             'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
             'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
             'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]

# YOUR CODE HERE
point_array = [4, 2, 1]
df = DataFrame({'countries':countries, 'gold':gold, 'silver':silver, 'bronze':bronze})
medals = df[['gold', 'silver', 'bronze']]
points = numpy.dot(medals, point_array)

olympic_points_df = DataFrame({'country_name':countries, 'points':points})





ohe = OneHotEncoder()
label_encoder = LabelEncoder()

data_label_encoded = label_encoder.fit_transform(df['countries'])
df['countries'] = data_label_encoded
print df

data_feature_one_hot_encoded = ohe.fit_transform(df[['countries']].as_matrix())
print data_feature_one_hot_encoded