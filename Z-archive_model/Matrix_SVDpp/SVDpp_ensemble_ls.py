from statistics import mode
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from dataset import extract_users_movies_ratings_lists, TripletDataset, save_predictions
import torch 

#Useful constants
number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 58
DATA_DIR = '../data'
EXPERIMENT_NAME = 'SVDpp_ensemble_ls'
import uuid
EXPERIMENT_NAME+='-'+str(uuid.uuid4())[:8]

N_TRIALS = 10000
import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'res_optuna/'
import os
os.makedirs(DIR_RESULTS+EXPERIMENT_NAME)

data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


import time
import pandas as pd
from libreco.data import split_by_ratio_chrono, DatasetPure
from libreco.algorithms import SVDpp
# remove unnecessary tensorflow logging
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)

test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
users_test, movies_test, ratings_test = extract_users_movies_ratings_lists(test_pd)

train = pd.DataFrame({'user': users_train, 'item': movies_train, 'label': ratings_train})
val = pd.DataFrame({'user': users_val, 'item': movies_val, 'label': ratings_val})

train_data, data_info = DatasetPure.build_trainset(train)
eval_data = DatasetPure.build_evalset(val)

start = 4
end = 40
best_epochs = [2, 2, 2, 2, 2, 5, 2, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

def prepate_model(emb_size):
    with tf.compat.v1.variable_scope(f'model_{emb_size}'):
        svdpp = SVDpp(task="rating", data_info=data_info, embed_size=emb_size,
                        n_epochs=best_epochs[emb_size-start], lr=0.001, reg=None, batch_size=256)
        svdpp.fit(train_data, verbose=2, eval_data=eval_data,
                    metrics=["rmse", "mae", "r2"])
        yhat = svdpp.predict(user=users_val, item=movies_val)
        del svdpp
        return yhat


import gc
val_yhat = []
for i in range(start, end+1):
    yhat = prepate_model(i)
    val_yhat.append(yhat)
    gc.collect()



val_base_model = np.column_stack(val_yhat)


coeffs = np.linalg.lstsq(val_base_model, ratings_val)[0]

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1)
ridge.fit(val_base_model, ratings_val)
coeffs_ridge = ridge.coef_

#Prediction on whole dataset
users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(data_pd)
train = pd.DataFrame({'user': users_train, 'item': movies_train, 'label': ratings_train})
train_data, data_info = DatasetPure.build_trainset(train)

test_yhat = []
for i in range(len(best_epochs)):
    with tf.compat.v1.variable_scope(f'model_full{i}'):
        svdpp = SVDpp(task="rating", data_info=data_info, embed_size=start+i,
                        n_epochs=best_epochs[i], lr=0.001, reg=None, batch_size=256)
        svdpp.fit(train_data, verbose=2, eval_data=eval_data,
                    metrics=["rmse", "mae", "r2"])
        yhat = svdpp.predict(user=users_test, item=movies_test)
        test_yhat.append(yhat)
        del svdpp
        gc.collect()

test_base_model = np.column_stack(test_yhat)

pred = np.matmul(test_base_model, coeffs)

save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', pred)

pred = np.matmul(test_base_model, coeffs_ridge)
EXPERIMENT_NAME = 'SVDpp_ensemble_ridge'
save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', pred)


pred = np.mean(test_base_model, axis=1)
EXPERIMENT_NAME = 'SVDpp_ensemble_mean'
save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', pred)