import numpy as np
import pandas as pd
#======

import matplotlib.pyplot as plt

#==========
header = ['IP', 'Time', 'URL', 'Staus']
urls_df = pd.read_csv('weblog.csv', sep='\t', names=header)
print (urls_df)
header = ['user_id', 'webpage_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)

# shape
print(df.shape)

# Calculate mean rating of all webpages 
df.groupby('webpage_id')['rating'].mean().sort_values(ascending=False).head() 

# creating dataframe with 'rating' count values 
ratings = pd.DataFrame(df.groupby('webpage_id')['rating'].mean()) 

ratings['num of ratings'] = pd.DataFrame(df.groupby('webpage_id')['rating'].count()) 

ratings.head() 

2
# head
print(df.head(5))

1
2
# descriptions
print(df.describe())

# rating distribution
print(df.groupby('rating').size())

# histograms
df.hist()
plt.show()



n_users = df.user_id.unique().shape[0]
n_webpage = df.webpage_id.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of pages = ' + str(n_webpage)  )




from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df.values, test_size=.12)

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_webpage))
for line in train_data:
                train_data_matrix[line[0]-1, line[1]-1] = line[2]

test_data_matrix = np.zeros((n_users, n_webpage))
for line in test_data:
                test_data_matrix[line[0]-1, line[1]-1] = line[2]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')

def predict(ratings, similarity, type='user'):
                if type == 'user':
                                mean_user_rating = ratings.mean(axis=1)
                                ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
                                pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T   
                                return pred

user_prediction = predict(train_data_matrix, user_similarity, type='user')

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
                prediction = prediction[ground_truth.nonzero()].flatten() 
                ground_truth = ground_truth[ground_truth.nonzero()].flatten()
                
                return sqrt(mean_squared_error(prediction, ground_truth))
                
print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print ('Accurecy: ' + str(100-rmse(user_prediction, test_data_matrix)))
