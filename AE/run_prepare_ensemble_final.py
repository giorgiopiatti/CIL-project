import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger

from dataset import (DatasetComplete, extract_matrix_users_movies_ratings,
                     save_predictions_from_pandas)
from model import Model
from swa_model import SWAModel

DATA_DIR = '../data_val_train_kfold/'
number_of_users, number_of_movies = (10000, 1000)

import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR_RESULTS = os.getenv('BASE_DIR_RESULTS')
DIR_RESULTS=BASE_DIR_RESULTS+'results_ensemble/'
EXPERIMENT_NAME = 'AE'
NEPTUNE_LOG_OFFLINE = True
print(DIR_RESULTS)

os.makedirs(DIR_RESULTS+EXPERIMENT_NAME, exist_ok=True)


def help_extract_optuna_params(params):
    output = defaultdict(lambda: {})
    for k, v in params.items():
        prefix, suffix = k.split('__', 1)
        output[prefix][suffix] = v
    return output


global_best_params = {'base__hidden_size': 92,
                      'base__encoding_size': 250,
                      'base__z_p_dropout': 0.9234988744649322,
                      'base__lr': 0.000623438837,
                      'base__num_layers': 2,
                      'base__weight_decay': 0.00049754544443277,
                      'ensemble__lr_low': 0.00012432605981159723,
                      'ensemble__lr_high': 0.00027772318534326874,
                      'ensemble__weight_decay': 0.0008597354444154666,
                      'ensemble__frequency_step': 5,
                      'trainer__epochs_base': 40,
                      'trainer__num_ensemble_models': 16,
                      'trainer__batch_size': 256}

train_pd = pd.read_csv('../data/data_train.csv')
test_pd = pd.read_csv('../data/sampleSubmission.csv')

best_params = help_extract_optuna_params(global_best_params)
params = best_params['base']
params_ensemble = best_params['ensemble']
trainer_config = best_params['trainer']

num_steps_epoch = math.ceil(number_of_users / trainer_config['batch_size'])


matrix_users_movies_train, _ = extract_matrix_users_movies_ratings(train_pd)
users_centered = []
users_mean = []
users_std = []
for i in range(number_of_users):
    user = matrix_users_movies_train[i]
    mean = user[user != 0].mean()
    std = user[user != 0].std()

    centered_user = np.where(user == 0, user, (user-mean)/std)
    users_centered.append(centered_user)
    users_mean.append(mean)
    users_std.append(std)


matrix_users_movies_train = np.stack(users_centered)
users_mean_train = np.stack(users_mean)
users_std_train = np.stack(users_std)

d_train = DatasetComplete(matrix_users_movies_train)
train_dataloader = torch.utils.data.DataLoader(
    d_train, batch_size=trainer_config['batch_size'], drop_last=True, shuffle=True)

d_test = DatasetComplete(matrix_users_movies_train)
test_dataloader = torch.utils.data.DataLoader(
    d_test, batch_size=trainer_config['batch_size'], drop_last=False, shuffle=False)


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


model = Model(
    users_mean=users_mean_train,
    users_std=users_std_train, **params)

# Base model training
trainer = pl.Trainer(
    max_epochs=trainer_config['epochs_base'],
    accelerator="gpu" if torch.cuda.is_available() else None,
    devices=1,
    log_every_n_steps=1,
    detect_anomaly=True,
    track_grad_norm=2,
    enable_progress_bar=True,
    callbacks=[],
    logger=neptune_logger,
    default_root_dir=DIR_RESULTS,
)
trainer.fit(model, train_dataloaders=train_dataloader)

# SWA
trainer.save_checkpoint(f'{DIR_RESULTS}/{EXPERIMENT_NAME}/AE_base_weights.ckpt')
model = Model.load_from_checkpoint(
    checkpoint_path=f'{DIR_RESULTS}/{EXPERIMENT_NAME}/AE_base_weights.ckpt', **params)

# SWA
swa = SWAModel(model, lr_low=params_ensemble['lr_low'], lr_high=params_ensemble['lr_high'], weight_decay=params_ensemble['weight_decay'],
               frequency_step=params_ensemble['frequency_step']*num_steps_epoch)

trainer = pl.Trainer(
    max_epochs=params_ensemble['frequency_step']*trainer_config['num_ensemble_models'],
    accelerator="gpu" if torch.cuda.is_available() else None,
    devices=1,
    log_every_n_steps=1,
    detect_anomaly=True,
    enable_progress_bar=False,
    default_root_dir=DIR_RESULTS,
    logger=neptune_logger,
    callbacks=[
        LearningRateMonitor(logging_interval='step')]
)
trainer.fit(swa, train_dataloaders=train_dataloader)

predictions = trainer.predict(swa, dataloaders=test_dataloader)
yhat = torch.concat(predictions)

save_predictions_from_pandas(
    f'{DIR_RESULTS}/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_final_results.csv', yhat, test_pd)
