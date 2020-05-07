import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

print("Downloading movielens data...")
from urllib.request import urlretrieve
import zipfile
urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip")
zip_ref = zipfile.ZipFile('movielens.zip', "r")
zip_ref.extractall()

rating_col = ['userId','movieId','rating','timestamp']
ratings_df = pd.read_csv("/content/ml-1m/ratings.dat",'::',names=rating_col,engine='python')
print(ratings_df.head())
#  ratings_df = ratings_df.sample(frac=0.01)

n_users = ratings_df.userId.unique().shape[0]
n_items = ratings_df.movieId.unique().shape[0]
print ('Total Users : '+ str(n_users))
print('Total Movie : ' + str(n_items))

ratings_utilitymatrix = ratings_df.pivot(index = 'userId',columns='movieId',values = 'rating').fillna(0)
print(ratings_utilitymatrix.head())



def domain_split(ratings,frac):
  users = ratings.shape[0]
  d1users = int(frac*users)
  d1 = ratings[0:d1users,:]
  d2 = ratings[d1users:users,:]
  return d1,d2

ratings = np.array(ratings_utilitymatrix)
no_ratings = np.count_nonzero(ratings)
print('Domain    Ratings    Users    Movies')
print('D          '+str(no_ratings) +'    '+str(ratings.shape[0]) + '    '+str(ratings.shape[1]))

ds,dt = domain_split(ratings,0.7)

print('Ds         '+str(np.count_nonzero(ds)) +'    '+str(ds.shape[0]) + '    '+str(ds.shape[1]))
print('Dt         '+str(np.count_nonzero(dt)) +'    '+str(dt.shape[0]) + '    '+str(dt.shape[1]))

def train_test_split(ratings,fractionTest):
	test = np.zeros(ratings.shape)
	train = ratings.copy()
	for user in range(ratings.shape[0]):
		nonzeroarr = ratings[user,:].nonzero()[0]
		#  print(nonzeroarr[0])
		nonzerolen = len(nonzeroarr)
		#  print(nonzerolen)
		test_rating_indices = np.random.choice(nonzeroarr,size=int(nonzerolen*fractionTest),replace=False)
		train[user,test_rating_indices] = 0
		test[user,test_rating_indices] = ratings[user,test_rating_indices]

	assert(np.all((train*test)==0))
	return train,test

dt_train,dt_test = train_test_split(dt,0.3)
dt_test,dt_valid = train_test_split(dt_test,0.33)

nets = np.concatenate((ds,dt_train))
print(nets.shape)

def matrix_factorization(R, P, Q, K, steps=1, alpha=0.0002, beta=0.02):
    Qtrans = Q.T
    for step in range(steps):

        eij = np.dot(P,Qtrans)
        eij = R - eij
        eij[eij<0] = 0
        tempq = Q

        for i in range(len(R)):
            P[i] = P[i] - alpha*beta*P[i]
            for j in range(len(R[i])):
                
                tempq[j] = 2*Q[j]*eij[i][j]*alpha

                for k in range(K):
                        #  w:=w-Delta(w)
                        #  Delta(p[i][k]) = -2*eij*Q[k][j]
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

nets_train,nets_test = train_test_split(nets,0.3)
print(nets_train.shape)
print(nets_test.shape)
nets_mean = np.mean(nets_train, axis = 1)
nets_demeaned = nets_train - nets_mean.reshape(-1, 1)
#  print(R_demeaned)
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(nets_demeaned, k = 25)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + nets_mean.reshape(-1, 1)

print(all_user_predicted_ratings.shape)
print(nets.shape)

from sklearn.metrics import mean_squared_error
def get_mse(pred, actual):
    #  Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

print(sqrt(get_mse(all_user_predicted_ratings,nets_test)))



