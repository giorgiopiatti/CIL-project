import os
import sys

import numpy as np
import pandas as pd
from libreco.algorithms import ALS
from libreco.data import DatasetPure

from dataset import (extract_users_movies_ratings_lists, save_predictions,
                     save_predictions_from_pandas)

import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'res_ensemble/'
EXPERIMENT_NAME = 'ALS_ensemble'

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME, exist_ok=True)

split_id = int(sys.argv[1])

test_pd = pd.read_csv('../data/sampleSubmission.csv')
users_test, movies_test, _ = extract_users_movies_ratings_lists(test_pd)

train_pd = pd.read_csv(f'../data_val_train_kfold/partition_{split_id}_train.csv')
val_pd = pd.read_csv(f'../data_val_train_kfold/partition_{split_id}_val.csv')
users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)

als_train_data, als_data_info = DatasetPure.build_trainset(pd.DataFrame(
    {'user': users_train, 'item': movies_train, 'label': ratings_train}))

emb_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 32, 58]
regs = [0.1, 0.5, 1, 5, 10]
ensemble_results_matrix_val = np.zeros((len(emb_sizes) * len(regs), len(users_val)))
ensemble_results_matrix_test = np.zeros((len(emb_sizes) * len(regs), len(users_test)))

for i, emb_size in enumerate(emb_sizes):
    for j, reg in enumerate(regs):
        base_model = ALS(task="rating", data_info=als_data_info, embed_size=emb_size, n_epochs=5,
                         reg=reg, seed=42)
        base_model.fit(als_train_data, verbose=0, use_cg=False,
                       n_threads=8, metrics=["rmse", "mae", "r2"])
        val_res = np.array(base_model.predict(user=users_val, item=movies_val))
        test_res = np.array(base_model.predict(user=users_test, item=movies_test))
        ensemble_results_matrix_val[len(regs)*i+j] = val_res
        ensemble_results_matrix_test[len(regs)*i+j] = test_res

yhat_val = ensemble_results_matrix_val.mean(axis=0)
yhat_test = ensemble_results_matrix_test.mean(axis=0)
save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{split_id}_val_results.csv', yhat_val, val_pd)
save_predictions(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{split_id}_test_results.csv', yhat_test)
