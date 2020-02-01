import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
names = ["userId","movieId","rating","timestamp"]
df = pd.read_csv("ratings.csv",names=names,header=0)
train_df = pd.DataFrame(columns=names)
test_df = pd.DataFrame(columns=names)
cut = 0.70
grouped = df.groupby('userId')
for name,group in grouped:
        X = group[['userId','movieId','timestamp']]
        Y = group[['rating']]
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=cut)
        X_train = pd.concat([X_train,Y_train],axis = 1)
        train_df = train_df.append(X_train,sort=False)
        X_test = pd.concat([X_test,Y_test],axis=1)
        test_df = test_df.append(X_test,sort=False)
print(test_df.shape)
test_df.to_csv("test.csv")
train_df.to_csv("train.csv")
