from scipy import stats
from libreco.data import DatasetPure
from libreco.algorithms import SVDpp
import tensorflow as tf
import gc
import os
import sys

import numpy as np
import pandas as pd

from dataset import (extract_users_movies_ratings_lists,
                     save_predictions_from_pandas)

k = int(sys.argv[1])


# Useful constants
number_of_users, number_of_movies = (10000, 1000)
EXPERIMENT_NAME = 'SVDpp_ensemble'

import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'results_ensemble/'

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME, exist_ok=True)


DATA_DIR = '../data_val_train_kfold/'
# Data source and split into val and train
train_pd = pd.read_csv(DATA_DIR+f'partition_{k}_train.csv')
val_pd = pd.read_csv(DATA_DIR+f'partition_{k}_val.csv')

users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)

# Data source and split into val and train
test_pd = pd.read_csv('../data/sampleSubmission.csv')
users_test, movies_test, ratings_test = extract_users_movies_ratings_lists(test_pd)


# remove unnecessary tensorflow logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

train = pd.DataFrame({'user': users_train, 'item': movies_train, 'label': ratings_train})
train_data, data_info = DatasetPure.build_trainset(train)

start = 4
end = 40
# Best epochs are found via SVDpp_ensemble_optimal_epochs.py
best_epochs = [2, 2, 2, 2, 2, 5, 2, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3,
               2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

best_params = {'mu': 39.9977395382687, 'sigma': 19.297573748183872}


def combine_models(mu, sigma, yhat):
    coeff = np.linspace(start, end, num=(end-start+1))
    coeff = stats.norm.pdf(coeff, loc=mu, scale=sigma)
    coeff = coeff / coeff.sum()
    return np.matmul(yhat, coeff)


test_yhat = []
val_yhat = []

for i in range(len(best_epochs)):
    with tf.compat.v1.variable_scope(f'model_full{i}'):
        svdpp = SVDpp(task="rating", data_info=data_info, embed_size=start+i,
                      n_epochs=best_epochs[i], lr=0.001, reg=None, batch_size=256)
        svdpp.fit(train_data, verbose=2,
                  metrics=["rmse", "mae", "r2"])
        yhat = svdpp.predict(user=users_test, item=movies_test)
        test_yhat.append(yhat)

        b = svdpp.predict(user=users_val, item=movies_val)
        val_yhat.append(b)
        del svdpp
        gc.collect()


test_base_model = np.column_stack(test_yhat)
val_base_model = np.column_stack(val_yhat)

test_pred = combine_models(best_params['mu'], best_params['sigma'], test_base_model)
val_pred = combine_models(best_params['mu'], best_params['sigma'], val_base_model)

save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{k}_test_results.csv', test_pred, test_pd)
save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{k}_val_results.csv', val_pred, val_pd)
