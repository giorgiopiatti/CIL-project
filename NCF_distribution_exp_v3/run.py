#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from model import NCFDistribution
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.model_selection import train_test_split
from dataset import extract_users_movies_ratings_lists, TripletDataset, save_predictions
import numpy as np


# In[2]:


nn = NCFDistribution(
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
    loss_use_class_weights=False
)


# In[35]:


df = pd.read_csv('./ratings.csv')
df = df[['userId', 'movieId', 'rating']]
df = df.loc[df['rating'] % 1 == 0]
df['rating'] = df['rating'].astype(np.int)
df['userId'] = df['userId'].map(lambda x: x-1)
df['movieId'] = df['movieId'].map(lambda x: x-1)
df['rating'] = df['rating'].map(lambda x: x-1)


# In[22]:


RANDOM_STATE = 58
BATCH_SIZE = 256


train_pd, val_pd = train_test_split(df, train_size=0.9, random_state=RANDOM_STATE)
users_train, movies_train, ratings_train = train_pd['userId'].values, train_pd['movieId'].values, train_pd['rating'].values

d_train = TripletDataset(users_train, movies_train, ratings_train)
train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

users_val, movies_val, ratings_val = val_pd['userId'].values, val_pd['movieId'].values, val_pd['rating'].values
d_val= TripletDataset(users_val, movies_val, ratings_val)
val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


# In[37]:


trainer = pl.Trainer(
        max_epochs=20, 
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1, 
        log_every_n_steps=1, 
        detect_anomaly=True, 
        track_grad_norm=2,
        )
trainer.fit(nn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

trainer.save_checkpoint('/cluster/scratch/dgu/CIL/NCF_distribution_exp_v3/NN/NN_final_weights.ckpt')

