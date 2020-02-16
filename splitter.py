import numpy as np 
import pandas as pd 
import sys
from math import sqrt
from sklearn.metrics import mean_squared_error
# df is dataframe
df = pd.read_csv('ratings.csv', ',')
print(df.head())


n_users = df.userId.unique().shape[0]
n_items = df.movieId.unique().shape[0]
print ('Total Users : '+ str(n_users))
print('Total Movie : ' + str(n_items))


# # A[:,0] first column of matrix A
# # A[:,2:5] All rows column 2nd to 5th



# converting data frame to utility matrix .
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
		sim_matrix[user,nonzeroarr] = mat[user,nonzeroarr] - avg + 1e-9
	
	sim_matrix = (sim_matrix).dot((sim_matrix).T)
	norms = np.array([np.sqrt(np.diagonal(np.abs(sim_matrix)))])
	# norms is a square root array of magnitude of each user (diagonal contains magntitude of rows)
	return (sim_matrix / norms / norms.T)

def cosine_sim(ratings, epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    sim = ratings.dot(ratings.T) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

sim_cos = cosine_sim(train)
sim_matrix = pearson_sim(train)

print('Similarity Matrix Calulated')
# print(sim_cos)
# print(sim_matrix)

def predict(ratings,similarity):
	# Summation sim(u,u')*r(u',i) / Summation of |sim(u,u')|
	den = np.array(np.abs(similarity).sum(axis=1)).T
	den = den.reshape((den.shape[0],1))
	return similarity.dot(ratings) /den

def predict_nobias(ratings,similarity):
	# r(ui) = ravg(u) + sum((sim(u,u')*(ru'i - ravgu'))/sum(sim(u,u'))
	user_bias = ratings.mean(axis=1)
	ratings = (ratings - user_bias[:, np.newaxis]).copy()
	pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
	pred += user_bias[:, np.newaxis]
	return pred

def predict_topk(ratings,similarity,k=40,bias=0):
    # make new similarity matrix having only top k similar users
    pred = np.zeros(ratings.shape)
    new_sim = np.zeros(similarity.shape)
    
    for i in range(similarity.shape[0]):
        x = tuple(np.sort(np.argsort(similarity[:,i])[:-k-1:-1]))
        new_sim[i][[x]] = similarity[i][[x]]
        
    if bias is 1:
        return predict(ratings,new_sim)
    return predict_nobias(ratings,new_sim)

def get_rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


user_prediction_cos_bias = predict(train, sim_cos)
print ('User-based CF MSE:(Cosine) ' + str(get_rmse(user_prediction_cos_bias, test)))
user_prediction_pearson_bias = predict(train,sim_matrix)
print ('User-based CF MSE:(Pearson)' + str(get_rmse(user_prediction_pearson_bias, test)))

user_prediction_cos_nobias = predict_nobias(train, sim_cos)
print ('User-based CF MSE:(Cosine) ' + str(get_rmse(user_prediction_cos_nobias, test)))
user_prediction_pearson_nobias = predict_nobias(train,sim_matrix)
print ('User-based CF MSE:(Pearson)' + str(get_rmse(user_prediction_pearson_nobias, test)))

user_prediction_cos_bias_topk = predict_topk(train, sim_cos,bias=1)
print ('User-based CF MSE:(Cosine,TOP-K) ' + str(get_rmse(user_prediction_cos_bias_topk, test)))
user_prediction_pearson_bias_topk = predict_topk(train,sim_matrix,bias=1)
print ('User-based CF MSE:(Pearson,TOP-K) ' + str(get_rmse(user_prediction_pearson_bias_topk, test)))

user_prediction_cos_nobias_topk = predict_topk(train, sim_cos)
print ('User-based CF MSE:(Cosine,TOP-K) without bias ' + str(get_rmse(user_prediction_cos_nobias_topk, test)))
user_prediction_pearson_nobias_topk = predict_topk(train,sim_matrix)
print ('User-based CF MSE:(Pearson,TOP-K) without bias ' + str(get_rmse(user_prediction_pearson_nobias_topk, test)))

#we consider cosine top-k matrix since it's our best result so far
# we divide our data into ranges (0-1],(1-2].....etc to create confusion matrix
prediction_range_values = np.ceil(abs(user_prediction_cos_nobias_topk)).flatten()
rating_range_values = np.ceil(test).flatten()
cf_matrix = pd.crosstab(pd.Series(rating_range_values,name='Actual'),pd.Series(prediction_range_values,name='Predicted'))
print(cf_matrix[1:])
