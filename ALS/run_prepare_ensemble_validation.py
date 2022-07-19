import importlib
import dataset
importlib.reload(dataset)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from dataset import extract_matrix_users_movies_ratings, extract_users_movies_ratings_lists, save_predictions, save_predictions_from_pandas
from libreco.data import split_by_ratio_chrono, DatasetPure
from libreco.algorithms import ALS
from sklearn.metrics import mean_squared_error


DIR_RESULTS = '/cluster/scratch/ncorecco/CIL/res_sub_ensemble/'
EXPERIMENT_NAME = f'ALS'
import os
os.makedirs(DIR_RESULTS+EXPERIMENT_NAME, exist_ok=True)

import sys
i = int(sys.argv[1])

def eval_ensemble(split, train_data, data_info, users_val, movies_val, ratings_val, users_test, movies_test):
    emb_sizes = [2,3,4,5,6,7,8,9,10,11,16,32,58]
    regs = [0.1, 0.5, 1, 5, 10]
    alphas = [0.01, 0.1, 1, 5, 8, 10, 100]
    david_val = np.zeros((len(emb_sizes) * len(regs) * len(alphas), len(users_val)))
    david_test = np.zeros((len(emb_sizes) * len(regs) * len(alphas), len(users_test)))

    for i, emb_size in enumerate(emb_sizes):
        for j, reg in enumerate(regs):
            for k, alpha in enumerate(alphas):
                giorgio = ALS(task="rating", data_info=data_info, embed_size=emb_size, n_epochs=5,
                            reg=reg, alpha=alpha, seed=42)
                giorgio.fit(train_data, verbose=0, use_cg=False, n_threads=8, metrics=["rmse", "mae", "r2"])
                david_val[len(regs)*len(alphas)*i+len(alphas)*j+k] = np.array(giorgio.predict(user=users_val, item=movies_val))
                david_test[len(regs)*len(alphas)*i+len(alphas)*j+k] = np.array(giorgio.predict(user=users_test, item=movies_test))

    yhat_val = david_val.mean(axis=0)
    yhat_test = david_test.mean(axis=0)
    save_predictions_from_pandas(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/ALS_split_{split}_val_results.csv', yhat_val, val_pd)
    save_predictions(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/ALS_split_{split}_test_results.csv', yhat_test)
    
    return mean_squared_error(yhat_val, ratings_val, squared=False)

test_pd = pd.read_csv('../data/sampleSubmission.csv')
users_test, movies_test, _ = extract_users_movies_ratings_lists(test_pd)


train_pd = pd.read_csv(f'../data_val_train_kfold/partition_{i}_train.csv')
val_pd = pd.read_csv(f'../data_val_train_kfold/partition_{i}_val.csv')
users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
train = pd.DataFrame({'user': users_train, 'item': movies_train, 'label': ratings_train})
val = pd.DataFrame({'user': users_val, 'item': movies_val, 'label': ratings_val})
train_data, data_info = DatasetPure.build_trainset(train)
eval_data = DatasetPure.build_evalset(val)

eval_ensemble(i, train_data, data_info, users_val, movies_val, ratings_val, users_test, movies_test)
