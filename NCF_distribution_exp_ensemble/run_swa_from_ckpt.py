from pyro import param
import pytorch_lightning as pl
import torch
import pandas as pd
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor
from neptune.new.types import File
from sklearn.model_selection import train_test_split

from dataset import extract_users_movies_ratings_lists, TripletDataset, save_predictions

#Useful constants
number_of_users, number_of_movies = (10000, 1000)
RANDOM_STATE = 42
BATCH_SIZE = 256
DATA_DIR = '../data'

#Data source and split into val and train
data_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
train_pd, val_pd = train_test_split(data_pd, train_size=0.9, random_state=RANDOM_STATE)


users_train, movies_train, ratings_train = extract_users_movies_ratings_lists(train_pd)
d_train = TripletDataset(users_train, movies_train, ratings_train)
train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

users_val, movies_val, ratings_val = extract_users_movies_ratings_lists(val_pd)
d_val= TripletDataset(users_val, movies_val, ratings_val)
val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
users_test, movies_test, ratings_test = extract_users_movies_ratings_lists(test_pd)
d_test= TripletDataset(users_test, movies_test, ratings_test, is_test_dataset=True)
test_dataloader = torch.utils.data.DataLoader(d_test, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)



EXPERIMENT_NAME = 'NCF_dist_exp_ensemble manual SWA'
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

from model import NCFDistribution
from swa_model import SWAModel

# Trial 137 finished with value: 0.9860949516296387 and parameters: 
params = {'embedding_size': 43, 'hidden_size': 117, 'alpha': 0.30184561739442606, 'sigma_prior': 0.5280354660742225, 'distance_0_to_3': 0.845472089209157, 'distance_3_to_2': 1.0123683337747076, 'distance_2_to_1': 0.12520765022811642, 'distance_0_to_4': 0.24389896700863054, 'distance_4_to_5': 1.9232424230681977, 'p_dropout': 0.14010135653155792, 
'weight_decay': 7.594599314482437e-05, 'scaling': 2.5967376547477308}

model = NCFDistribution.load_from_checkpoint('./end_training.ckpt', **params)


params_opt = {
    'lr_low' : 5e-4,
    'lr_high' : 1e-3,
    'weight_decay': 7.594599314482437e-05,
    'frequency_step' : 3
}
NUM_STEPS_EPOCH = 4137
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