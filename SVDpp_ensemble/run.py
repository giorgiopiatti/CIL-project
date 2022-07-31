import os

import pandas as pd
import tensorflow as tf
from libreco.algorithms import SVDpp
from libreco.data import DatasetPure
from sklearn.model_selection import train_test_split

from dataset import (extract_users_movies_ratings_lists,
                     save_predictions)

# Useful constants
number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 58
DATA_DIR = '../data'
EXPERIMENT_NAME = 'SVDpp reg'
data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


# remove unnecessary tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# %%
users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)

test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
users_test, movies_test, ratings_test = extract_users_movies_ratings_lists(test_pd)

train = pd.DataFrame({'user': users_train, 'item': movies_train, 'label': ratings_train})
val = pd.DataFrame({'user': users_val, 'item': movies_val, 'label': ratings_val})

# %%
train_data, data_info = DatasetPure.build_trainset(train)
eval_data = DatasetPure.build_evalset(val)

# %%
svdpp = SVDpp(task="rating", data_info=data_info, embed_size=20,
              n_epochs=4, lr=0.001, reg=0.01, batch_size=256)
svdpp.fit(train_data, verbose=2, eval_data=eval_data,
          metrics=["rmse", "mae", "r2"])


yhat = svdpp.predict(user=users_test, item=movies_test)
save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', yhat)
