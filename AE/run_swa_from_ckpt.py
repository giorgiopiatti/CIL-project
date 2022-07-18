import pytorch_lightning as pl
import torch
import pandas as pd
from pytorch_lightning.loggers import NeptuneLogger
from neptune.new.types import File
from sklearn.model_selection import train_test_split
import numpy as np
from dataset import extract_matrix_users_movies_ratings, DatasetComplete, DatasetValidation, save_predictions
from pytorch_lightning.callbacks import LearningRateMonitor
#Useful constants
number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 58
BATCH_SIZE = 256
DATA_DIR = '../data'


#Data source and split into val and train
data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


matrix_users_movies_train, _ = extract_matrix_users_movies_ratings(train_pd)
users_centered = []
users_mean = []
users_std = []
for i in range(number_of_users):
    user = matrix_users_movies_train[i]
    mean = user[user != 0].mean()
    std = user[user != 0].std()
    
    centered_user = np.where(user==0, user, (user-mean)/std)
    users_centered.append(centered_user)
    users_mean.append(mean)
    users_std.append(std)


matrix_users_movies_train = np.stack(users_centered)
users_mean_train = np.stack(users_mean)
users_std_train = np.stack(users_std)

d_train = DatasetComplete(matrix_users_movies_train)
train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

matrix_users_movies_val, _ = extract_matrix_users_movies_ratings(val_pd)
d_val= DatasetValidation(matrix_users_movies_train, matrix_users_movies_val)
val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


d_test= DatasetComplete(matrix_users_movies_train)
test_dataloader = torch.utils.data.DataLoader(d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)



EXPERIMENT_NAME = 'AE swa'
DEBUG = False


proxies = {
'http': 'http://proxy.ethz.ch:3128',
'https': 'http://proxy.ethz.ch:3128',
}
neptune_logger = NeptuneLogger(
    project="TiCinesi/CIL-project", 
    api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMzQyZmQ3MS02OGM5LTQ2Y2EtOTEzNC03MjBjMzUyN2UzNDMifQ==',
    mode = 'debug' if DEBUG else 'async',
    name=EXPERIMENT_NAME,
    tags=[],  # optional
    proxies=proxies,
    source_files='**/*.py'
)


EPOCHS = 100
from model import Model
from swa_model import SWAModel


params = {'hidden_size': 49, 'encoding_size': 40, 'z_p_dropout': 0.5740915050728582, 'lr': 0.008343212707743271, 'num_layers': 2, 'weight_decay': 0.000502065454118506}

model = Model.load_from_checkpoint('./end_training.ckpt', **params,  users_mean=users_mean_train,
    users_std=users_std_train)


params_opt = {
    'lr_low' : 4e-3,
    'lr_high' : 8e-3,
    'weight_decay': 7.594599314482437e-05,
    'frequency_step' : 3
}
NUM_STEPS_EPOCH = 40
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
trainer.fit(swa, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

predictions = trainer.predict(swa, dataloaders=test_dataloader)

yhat = torch.concat(predictions)

save_predictions(f'{EXPERIMENT_NAME}-predictedSubmission.csv', yhat)
neptune_logger.experiment['results/end_model'].upload(File(f'{EXPERIMENT_NAME}-predictedSubmission.csv'))