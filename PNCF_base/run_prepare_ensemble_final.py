from model import NCFDistribution
import os
import pandas as pd
import pytorch_lightning as pl
import torch
from neptune.new.types import File
from pytorch_lightning.loggers import NeptuneLogger

from dataset import (TripletDataset, extract_users_movies_ratings_lists,
                     save_predictions_from_pandas)

# Useful constants
number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 153
BATCH_SIZE = 256
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


EXPERIMENT_NAME = 'PNCF_base'
import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'res_ensemble/'

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


params = {'embedding_size': 43, 'hidden_size': 117, 'alpha': 0.30184561739442606,
          'sigma_prior': 0.5280354660742225, 'distance_0_to_3': 0.845472089209157,
          'distance_3_to_2': 1.0123683337747076, 'distance_2_to_1': 0.12520765022811642,
          'distance_0_to_4': 0.24389896700863054, 'distance_4_to_5': 1.9232424230681977, 'p_dropout': 0.14010135653155792,
          'weight_decay': 7.594599314482437e-05, 'scaling': 2.5967376547477308}

model = NCFDistribution(**params)

trainer = pl.Trainer(
    max_epochs=20,
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
