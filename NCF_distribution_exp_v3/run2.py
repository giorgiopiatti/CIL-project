#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from model import NCFDistribution
from model2 import NCFDistribution2
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.model_selection import train_test_split
from dataset import extract_users_movies_ratings_lists, TripletDataset, save_predictions
import numpy as np


# In[2]:


nn = NCFDistribution.load_from_checkpoint('./epoch=19-step=1386440.ckpt', 
                                        embedding_size_movie=43, 
    embedding_size_user=43,
    hidden_size = 117,
    lr=1e-3, 
    alpha=0.3,
    sigma_prior=0.528,
    distance_0_to_3 = 0.845,
    distance_3_to_2 = 1.012,
    distance_2_to_1 = 0.125,
    distance_0_to_4 = 0.243,
    distance_4_to_5 = 1.923,
    scaling= 2.597,
    p_dropout=0.14,
    weight_decay=0,
    loss_use_class_weights=False)


# In[3]:


f_theta_nn = nn.prob_dist
network = NCFDistribution2(
    embedding_size_movie=43, 
    embedding_size_user=43,
    lr=1e-3, 
    alpha=0.3,
    sigma_prior=0.528,
    distance_0_to_3 = 0.845,
    distance_3_to_2 = 1.012,
    distance_2_to_1 = 0.125,
    distance_0_to_4 = 0.243,
    distance_4_to_5 = 1.923,
    scaling= 2.597,
    f_theta = f_theta_nn
)

# In[4]:


number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 58
BATCH_SIZE = 256
DATA_DIR = '../data'

#Data source and split into val and train
data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(data_pd)
#train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


#users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
d_train = TripletDataset(users_train, movies_train, ratings_train)
train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

#users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
#d_val= TripletDataset(users_val, movies_val, ratings_val)
#val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
users_test, movies_test, ratings_test = extract_users_movies_ratings_lists(test_pd)
d_test= TripletDataset(users_test, movies_test, ratings_test, is_test_dataset=True)
test_dataloader = torch.utils.data.DataLoader(d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


# In[5]:


trainer = pl.Trainer(
        max_epochs=2, 
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1, 
        log_every_n_steps=1, 
        detect_anomaly=True, 
        track_grad_norm=2,
        )
#trainer.fit(network, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.fit(network, train_dataloaders=train_dataloader)

predictions = trainer.predict(network, dataloaders=test_dataloader, ckpt_path='best')

yhat = torch.concat(predictions)

save_predictions('TransferNN-predictedSubmission.csv', yhat)
