import numpy as np 
import pandas as pd 
import sys

# df is dataframe
df = pd.read_csv('ml-latest-small/ratings.csv', ',')
print(df.head())


n_users = df.userId.unique().shape[0]
n_items = df.movieId.unique().shape[0]
print ('Total Users : '+ str(n_users))
print('Total Movie : ' + str(n_items))

# # A[:,0] first column of matrix A
# # A[:,2:5] All rows column 2nd to 5th


# converting data frame to utility matrix 
df_utility = df.pivot(index = 'userId',columns='movieId',values = 'rating').fillna(0)
print(df_utility.head())
ratings = df_utility.to_numpy()
print(ratings.shape)


# print(ratings)
# # approximately 1 user rates 20 movies in the given data set.
# # So test and train set can be divided based on the above param
# def train_test_split(ratings,size):
#     test = np.zeros(ratings.shape)
#     train = ratings.copy()
#     for user in range(ratings.shape[0]):
#         test_ratings = np.random.choice(ratings[user, :].nonzero()[0], size=size, replace=False)
#         train[user, test_ratings] = 0.
#         test[user, test_ratings] = ratings[user, test_ratings]
        
#     return train, test



# train,test = train_test_split(ratings,5)

# print(mean_rows.shape)
# def cosine_sim(utilitymat,type):
# 	mean_rows = np.mean(train,axis=1)
# 	mean_rows = mean_rows.reshape((mean_rows.shape[0],1))
# 	if(type=='user'):
# 		sim = train.dot(train.T)
# 	elif(type=='item')
# 		sim = train.T.dot(train)

	
