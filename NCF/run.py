import os
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.model_selection import train_test_split

from dataset import (TripletDataset, extract_users_movies_ratings_lists,
                     save_predictions_from_pandas)
from model import Model

# Useful constants
number_of_users, number_of_movies = (10000, 1000)
BATCH_SIZE = 1024
RANDOM_STATE = 42
DATA_DIR = '../data/'

# Data source and split into val and train
data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)

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


EXPERIMENT_NAME = 'NCF'
import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'results_baseline/'

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME, exist_ok=True)
NEPTUNE_LOG_OFFLINE = True

proxies = {
    'http': 'http://proxy.ethz.ch:3128',
    'https': 'http://proxy.ethz.ch:3128',
}
neptune_logger = NeptuneLogger(
    project="TiCinesi/CIL-project",
    
    mode='debug' if NEPTUNE_LOG_OFFLINE else 'async',
    name=EXPERIMENT_NAME,
    tags=[],  # optional
    proxies=proxies,
    source_files='**/*.py'
)


params = {'emb_size': 16, 'lr': 1e-3}

model = Model(**params)

trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu" if torch.cuda.is_available() else None,
    devices=1,
    log_every_n_steps=1,
    detect_anomaly=True,
    track_grad_norm=2,
    logger=neptune_logger
)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

predictions = trainer.predict(model, dataloaders=test_dataloader)
yhat = torch.concat(predictions)
save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}-predictionSubmission', yhat, test_pd)
