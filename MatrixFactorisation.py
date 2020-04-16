import numpy
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
df = pd.read_csv('ml-latest-small/ratings.csv', ',')
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

def matrix_factorization(R, P, Q, K, steps=1000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        # w:=w-Delta(w)
                        # Delta(p[i][k]) = -2*eij*Q[k][j]
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


R = train
N = len(R)
M = len(R[0])
K = 2
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)
nP, nQ = matrix_factorization(R, P, Q, K)
predict = numpy.dot(nP, nQ.T)

error = (predict,test)
print(error)

