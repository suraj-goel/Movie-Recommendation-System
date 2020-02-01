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
ratings = np.array(df_utility)
# ratings numpy utility matrix

# print(ratings[0,:].nonzero())
# .nonzero() gives an array of indices having non zero values


def train_test_split(ratings,fractionTest):
	test = np.zeros(ratings.shape)
	train = ratings.copy()
	for user in range(ratings.shape[0]):
		nonzeroarr = ratings[user,:].nonzero()[0]
		# print(nonzeroarr[0])
		nonzerolen = len(nonzeroarr)
		# print(nonzerolen)
		test_rating_indices = np.random.choice(nonzeroarr,size=int(nonzerolen*fractionTest),replace=False)
		train[user,test_rating_indices]=0
		test[user,test_rating_indices] = ratings[user,test_rating_indices]

	assert(np.all((train*test)==0))
	return train,test

train,test = train_test_split(ratings,0.25) 
print('Data splitted into Train,Test')

def pearson_sim(mat):
	sim_matrix = np.zeros((mat.shape[0],mat.shape[1]))
	mean = np.mean(mat,axis=1)
	for user in range(mat.shape[0]):
		nonzeroarr = mat[user,:].nonzero()[0]
		avg = np.sum(mat[user])/len(nonzeroarr)
		sim_matrix[user,nonzeroarr] = mat[user,nonzeroarr] - avg
		
	sim_matrix = (sim_matrix).dot((sim_matrix).T) 
	# check pearson after this
	norms = np.array([np.sqrt(np.diagonal(sim_matrix))])
	print(norms)
	return sim_matrix / norms / norms.T

def cosine_sim(ratings, epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    sim = ratings.dot(ratings.T) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

tmp = np.array([[1,3,0,0,5],[2,3,0,0,4]])
print(tmp)
sim_matrix = pearson_sim(tmp)
print(sim_matrix)
print('Similarity Matrix Calulated')
sim_cos = cosine_sim(tmp)
print(sim_cos)