from scipy import stats
from libreco.data import DatasetPure
from libreco.algorithms import SVDpp
import tensorflow as tf
import gc
import os
import numpy as np
import pandas as pd

from dataset import (extract_users_movies_ratings_lists,
                     save_predictions_from_pandas)

# Useful constants
number_of_users, number_of_movies = (10000, 1000)
DATA_DIR = '../data'

EXPERIMENT_NAME = 'SVDpp_ensemble_gaussian'

DIR_RESULTS = '/cluster/scratch/piattigi/CIL/res_ensemble/'

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME, exist_ok=True)

# Data source and split into val and train
data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(data_pd)
test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
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
for i in range(len(best_epochs)):
    with tf.compat.v1.variable_scope(f'model_full{i}'):
        svdpp = SVDpp(task="rating", data_info=data_info, embed_size=start+i,
                      n_epochs=best_epochs[i], lr=0.001, reg=None, batch_size=256)
        svdpp.fit(train_data, verbose=2,
                  metrics=["rmse", "mae", "r2"])
        yhat = svdpp.predict(user=users_test, item=movies_test)
        test_yhat.append(yhat)
        del svdpp
        gc.collect()


test_base_model = np.column_stack(test_yhat)
pred = combine_models(best_params['mu'], best_params['sigma'], test_base_model)

save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_final_results.csv', yhat, test_pd)
