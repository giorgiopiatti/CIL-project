from swa_model import SWAModel
import math
from model import NCFDistribution
import os
import sys
import pandas as pd
import pytorch_lightning as pl
import torch
from neptune.new.types import File
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger

from dataset import (TripletDataset, extract_users_movies_ratings_lists,
                     save_predictions_from_pandas)

k = int(sys.argv[1])


# Useful constants
number_of_users, number_of_movies = (10000, 1000)
BATCH_SIZE = 256
DATA_DIR = '../data_val_train_kfold/'

# Data source and split into val and train
train_pd = pd.read_csv(DATA_DIR+f'partition_{k}_train.csv')
val_pd = pd.read_csv(DATA_DIR+f'partition_{k}_val.csv')

users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
d_train = TripletDataset(users_train, movies_train, ratings_train)
train_dataloader = torch.utils.data.DataLoader(
    d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
d_val = TripletDataset(users_val, movies_val, ratings_val)
val_dataloader = torch.utils.data.DataLoader(
    d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)

d_val_predict = TripletDataset(users_val, movies_val, ratings_val, is_test_dataset=True)
val_predict_dataloader = torch.utils.data.DataLoader(
    d_val_predict, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


test_pd = pd.read_csv('../data/sampleSubmission.csv')
users_test, movies_test, ratings_test = extract_users_movies_ratings_lists(test_pd)
d_test = TripletDataset(users_test, movies_test, ratings_test, is_test_dataset=True)
test_dataloader = torch.utils.data.DataLoader(
    d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


EXPERIMENT_NAME = 'PNCF'
import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'res_ensemble/'

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


params = {'embedding_size': 43,
          'hidden_size': 117,
          'alpha': 0.30184561739442606,
          'sigma_prior': 0.5280354660742225,
          'distance_0_to_3': 0.845472089209157,
          'distance_3_to_2': 1.0123683337747076,
          'distance_2_to_1': 0.12520765022811642,
          'distance_0_to_4': 0.24389896700863054,
          'distance_4_to_5': 1.9232424230681977,
          'p_dropout': 0.14010135653155792,
          'weight_decay': 7.594599314482437e-05,
          'scaling': 2.5967376547477308}

model = NCFDistribution.load_from_checkpoint(
    f'{DIR_RESULTS}/PNCF_base/PNCF_base_split_{k}_weights.ckpt', **params)

params_opt = {
    'lr_low': 5e-4,
    'lr_high': 1e-3,
    'weight_decay': 7.594599314482437e-05,
    'frequency_step': 3
}


NUM_STEPS_EPOCH = math.ceil(len(users_train) / BATCH_SIZE)

swa = SWAModel(model, lr_low=params_opt['lr_low'], lr_high=params_opt['lr_high'], weight_decay=params_opt['weight_decay'],
               frequency_step=params_opt['frequency_step']*NUM_STEPS_EPOCH)

trainer = pl.Trainer(
    max_epochs=params_opt['frequency_step']*10,
    accelerator="gpu" if torch.cuda.is_available() else None,
    devices=1,
    log_every_n_steps=1,
    detect_anomaly=True,
    track_grad_norm=2,
    logger=neptune_logger,
    callbacks=[
        LearningRateMonitor(logging_interval='step')]
)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.save_checkpoint(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{k}_weights.ckpt')

predictions = trainer.predict(model, dataloaders=test_dataloader)
yhat = torch.concat(predictions)
save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{k}_test_results.csv', yhat, test_pd)

predictions = trainer.predict(model, dataloaders=val_predict_dataloader)
yhat = torch.concat(predictions)
save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{k}_val_results.csv', yhat, val_pd)


neptune_logger.experiment[f'ensemble/{EXPERIMENT_NAME}_split_{k}_test_results'].upload(
    File(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{k}_test_results.csv'))
neptune_logger.experiment[f'ensemble/{EXPERIMENT_NAME}_split_{k}_val_results'].upload(
    File(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{k}_val_results.csv'))
neptune_logger.experiment[f'ensemble/{EXPERIMENT_NAME}_split_{k}_weights'].upload(
    File(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_split_{k}_weights.ckpt'))
