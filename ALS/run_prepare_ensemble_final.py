import importlib
import dataset
importlib.reload(dataset)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from dataset import extract_users_movies_ratings_lists, save_predictions
from libreco.data import split_by_ratio_chrono, DatasetPure
from libreco.algorithms import ALS
from sklearn.metrics import mean_squared_error


DIR_RESULTS = '/cluster/scratch/ncorecco/CIL/res_sub_ensemble/'
EXPERIMENT_NAME = f'ALS'
import os
os.makedirs(DIR_RESULTS+EXPERIMENT_NAME, exist_ok=True)


test_pd = pd.read_csv('../data/sampleSubmission.csv')
users_test, movies_test, _ = extract_users_movies_ratings_lists(test_pd)


train_pd = pd.read_csv(f'../data/data_train.csv')

users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)

train = pd.DataFrame({'user': users_train, 'item': movies_train, 'label': ratings_train})
train_data, data_info = DatasetPure.build_trainset(train)

# --------- generate submission -------
emb_sizes = [2,3,4,5,6,7,8,9,10,11,16,32,58]
regs = [0.1, 0.5, 1, 5, 10]
alphas = [0.01, 0.1, 1, 5, 8, 10, 100]
david = np.zeros((len(emb_sizes) * len(regs) * len(alphas), len(users_test)))
    
for i, emb_size in enumerate(emb_sizes):
    for j, reg in enumerate(regs):
        for k, alpha in enumerate(alphas):
            giorgio = ALS(task="rating", data_info=data_info, embed_size=emb_size, n_epochs=5,
                        reg=reg, alpha=alpha, seed=42)
            giorgio.fit(train_data, verbose=2, use_cg=False, n_threads=8, metrics=["rmse", "mae", "r2"])
            david[len(regs)*len(alphas)*i+len(alphas)*j+k] = np.array(giorgio.predict(user=users_test, item=movies_test))
            
yhat = david.mean(axis=0)
save_predictions(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_final_results.csv', yhat)