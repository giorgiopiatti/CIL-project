import os

import pandas as pd
import pytorch_lightning as pl
import torch
from neptune.new.types import File
from pytorch_lightning.loggers import NeptuneLogger

from dataset import (TripletDataset, extract_users_movies_ratings_lists,
                     save_predictions_from_pandas)
from model import Model

# Useful constants
number_of_users, number_of_movies = (10000, 1000)
BATCH_SIZE = 1024
DATA_DIR = '../data'

# Data source and split into val and train
data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')

users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(data_pd)
d_train = TripletDataset(users_train, movies_train, ratings_train)
train_dataloader = torch.utils.data.DataLoader(
    d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)


test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
users_test, movies_test, ratings_test = extract_users_movies_ratings_lists(test_pd)
d_test = TripletDataset(users_test, movies_test, ratings_test, is_test_dataset=True)
test_dataloader = torch.utils.data.DataLoader(
    d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


EXPERIMENT_NAME = 'NCF'
DIR_RESULTS = '/cluster/scratch/piattigi/CIL/res_baseline/'

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME, exist_ok=True)
DEBUG = False

proxies = {
    'http': 'http://proxy.ethz.ch:3128',
    'https': 'http://proxy.ethz.ch:3128',
}
neptune_logger = NeptuneLogger(
    project="TiCinesi/CIL-project",
    api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMzQyZmQ3MS02OGM5LTQ2Y2EtOTEzNC03MjBjMzUyN2UzNDMifQ==',
    mode='debug' if DEBUG else 'async',
    name=EXPERIMENT_NAME,
    tags=[],  # optional
    proxies=proxies,
    source_files='**/*.py'
)


params = {'emb_size': 16, 'lr': 1e-3}


model = Model(**params)

trainer = pl.Trainer(
    max_epochs=25,
    accelerator="gpu" if torch.cuda.is_available() else None,
    devices=1,
    log_every_n_steps=1,
    detect_anomaly=True,
    track_grad_norm=2,
    logger=neptune_logger
)
trainer.fit(model, train_dataloaders=train_dataloader)
trainer.save_checkpoint(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_final_weights.ckpt')

predictions = trainer.predict(model, dataloaders=test_dataloader)

yhat = torch.concat(predictions)

save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_final_results.csv', yhat, test_pd)
neptune_logger.experiment[f'ensemble/final_results'].upload(
    File(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_final_results.csv'))
neptune_logger.experiment[f'ensemble/final_weights'].upload(
    File(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_final_weights.ckpt'))
