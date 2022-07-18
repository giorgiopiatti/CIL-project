import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from dataset import extract_users_movies_ratings_lists, TripletDataset, save_predictions
import torch 

#Useful constants
number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 58
DATA_DIR = '../data'
EXPERIMENT_NAME = 'SVDpp'
N_TRIALS = 10

data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


import time
import pandas as pd
from libreco.data import split_by_ratio_chrono, DatasetPure
from libreco.algorithms import SVDpp
# remove unnecessary tensorflow logging
import os
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


import gc 

def prepate_model(emb_size, epochs):
    with tf.compat.v1.variable_scope(f'model_{emb_size}_{epochs}'):
        svdpp = SVDpp(task="rating", data_info=data_info, embed_size=emb_size,
                        n_epochs=epochs, lr=0.001, reg=None, batch_size=256)
        svdpp.fit(train_data, verbose=2, eval_data=eval_data,
                    metrics=["rmse", "mae", "r2"])
        yhat = svdpp.predict(user=users_val, item=movies_val)

        del svdpp
        return yhat

start = 4
end = 40


best_epochs = []
for i in range(start, end+1):
    loss_current = None
    best_epoch = None

    for epoch in range(2, 7):
        yhat = prepate_model(i, epoch)
        rmse = np.sqrt(np.mean((yhat-ratings_val)**2))

        if loss_current is None:
            loss_current = rmse
            best_epoch = epoch

        if rmse > loss_current:
            best_epoch = epoch -1
            break

    
    gc.collect()
    best_epochs.append(best_epoch)
    print('Emb_size ', i)
    print(best_epochs)


print(best_epochs)