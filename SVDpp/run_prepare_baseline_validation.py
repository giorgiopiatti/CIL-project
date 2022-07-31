import os
import sys

import numpy as np
import pandas as pd
from libreco.algorithms import ALS
from libreco.data import DatasetPure

from dataset import (extract_users_movies_ratings_lists,
                     save_predictions_from_pandas)

import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'results_baseline/'
EXPERIMENT_NAME = 'SVDpp'

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME, exist_ok=True)

split_id = int(sys.argv[1])

test_pd = pd.read_csv('../data/sampleSubmission.csv')
users_test, movies_test, _ = extract_users_movies_ratings_lists(test_pd)

train_pd = pd.read_csv(f'../data_val_train_kfold/partition_{split_id}_train.csv')
val_pd = pd.read_csv(f'../data_val_train_kfold/partition_{split_id}_val.csv')


def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

number_of_users, number_of_movies = (10000, 1000)
train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)

# also create full matrix of observed values
data = np.full((number_of_users, number_of_movies), np.mean(train_pd.Prediction.values))
mask = np.zeros((number_of_users, number_of_movies)) # 0 -> unobserved value, 1->observed value


train_users, train_movies, train_predictions = extract_users_movies_ratings_lists(train_pd)
test_users, test_movies, test_predictions = extract_users_movies_ratings_lists(val_pd)

for user, movie, pred in zip(train_users, train_movies, train_predictions):
    data[user - 1][movie - 1] = pred
    mask[user - 1][movie - 1] = 1

# choose how many singlular values to keep
k_singular_values = 2
number_of_singular_values = min(number_of_users, number_of_movies)

assert(k_singular_values <= number_of_singular_values), "choose correct number of singular values"

U, s, Vt = np.linalg.svd(data, full_matrices=False)

S = np.zeros((number_of_movies, number_of_movies))
S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])

reconstructed_matrix = U.dot(S).dot(Vt)

save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{split_id}_val_results.csv', reconstructed_matrix, val_pd)
save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{split_id}_test_results.csv', reconstructed_matrix, test_pd)
