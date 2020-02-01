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





	
