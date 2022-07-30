# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from dataset import extract_users_movies_ratings_lists, TripletDataset, extract_matrix_users_movies_ratings
import torch 

#Useful constants
number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 58
DATA_DIR = '../data'


data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


#users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
#users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
matrix_train, _ = extract_matrix_users_movies_ratings(train_pd)
matrix_val, _ = extract_matrix_users_movies_ratings(val_pd)

# %%
K=6
u = np.random.rand(number_of_users, K)
v = np.random.rand(number_of_movies, K)



# %%
def step(u, v, y):
    u_new = np.empty_like(u)
    v_new = np.empty_like(v)

    for i in range(0, number_of_users):
        for l in range(0, K):
            uv = np.matmul(u[i],v.transpose())
            a = v[:,l] * y[i, :]
            a = a / uv
            b = v[:, l].sum()
        
            u_new[i,l] = u[i,l]* (a.sum())/b
            
    
    for j in range(0, number_of_movies):
        for l in range(0, K):
            uv = np.matmul(v[j],u.transpose())
            a = u[:,l] * y[:, j]
            a = a / uv
            b = u[:, l].sum()
        
            v_new[j,l] = v[j,l]* (a.sum())/b
            
    return u_new, v_new


# %%
def rmse(y, yhat):
    mask = y.astype(np.bool)
    return np.sqrt(np.mean((y-yhat)[mask]**2))

# %%
old_score = np.inf
n = 0
while True:
    u, v = step(u, v, matrix_train)
    pred = np.matmul(u,v.transpose())
    val_score = rmse(matrix_val, pred)
    train_score = rmse(matrix_train, pred)

    print(f'Train rmse: {train_score}, val rmse: {val_score}')
    
    if n % 1000 == 0:
        if val_score < old_score:
            np.save('./both_best_u', u)
            np.save('./both_best_v', v)
            old_score = val_score
        np.save('./both_u', u)
        np.save('./both_v', v)
    n+=1



