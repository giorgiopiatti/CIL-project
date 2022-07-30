#@title Basic Imports

import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from ALS.dataset import extract_matrix_users_movies_ratings, extract_users_movies_ratings_lists

#@title Use GPU in colab: Runtime->Change Runtime type
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
    
DATA_DIR = 'data'
number_of_users, number_of_movies = (10000, 1000)


train_pd = pd.read_csv(DATA_DIR+'/data_train.csv')
test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')




rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))

def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

def save_predictions_from_pandas(res_path, predictions, index_pd):
    index_pd = index_pd.astype({'Prediction': 'float'})
    index_pd.iloc[:, 1] = predictions
    index_pd.to_csv(res_path, index=False, float_format='%.3f')

def save_predictions(res_path, predictions):
    test_pd = pd.read_csv('data/sampleSubmission.csv')
    test_pd = test_pd.astype({'Prediction': 'float'})
    test_pd.iloc[:, 1] = predictions
    test_pd.to_csv(res_path, index=False, float_format='%.3f')


#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

class NCF(nn.Module):
    def __init__(self, number_of_users, number_of_movies, embedding_size):
        super().__init__()
        self.embedding_layer_users = nn.Embedding(number_of_users, embedding_size)
        self.embedding_layer_movies = nn.Embedding(number_of_movies, embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=2 * embedding_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1), # maybe predict per category?
            nn.ReLU()
        )

    def forward(self, users, movies):
        users_embedding = self.embedding_layer_users(users)
        movies_embedding = self.embedding_layer_movies(movies)
        concat = torch.cat([users_embedding, movies_embedding], dim=1)
        return torch.squeeze(self.feed_forward(concat))

# Parameters
batch_size = 1024
num_epochs = 25
show_validation_score_every_epochs = 1
embedding_size = 16
learning_rate = 1e-3

def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


train_users, train_movies, train_predictions = extract_users_movies_ratings_lists(train_pd)
test_users, test_movies, test_predictions = extract_users_movies_ratings_lists(test_pd)

train_users_torch = torch.tensor(train_users, device=device)
train_movies_torch = torch.tensor(train_movies, device=device)
train_predictions_torch = torch.tensor(train_predictions, device=device)

test_users_torch = torch.tensor(test_users, device=device)
test_movies_torch = torch.tensor(test_movies, device=device)

train_dataloader = DataLoader(
TensorDataset(train_users_torch, train_movies_torch, train_predictions_torch),
batch_size=batch_size)

test_dataloader = DataLoader(
TensorDataset(test_users_torch, test_movies_torch),
batch_size=batch_size)

ncf = NCF(number_of_users, number_of_movies, embedding_size).to(device)
optimizer = optim.Adam(ncf.parameters(),
                       lr=learning_rate)

# TRAINING
step = 0
all_predictions = []
with tqdm(total=len(train_dataloader) * num_epochs) as pbar:
    for epoch in range(num_epochs):
        for users_batch, movies_batch, target_predictions_batch in train_dataloader:
            optimizer.zero_grad()

            predictions_batch = ncf(users_batch, movies_batch)

            loss = mse_loss(predictions_batch, target_predictions_batch)

            loss.backward()

            optimizer.step()

            pbar.update(1)
            step += 1

for users_batch, movies_batch in test_dataloader:
            predictions_batch = ncf(users_batch, movies_batch)
            all_predictions.append(predictions_batch)
                
all_predictions = torch.cat(all_predictions)
save_predictions('Normal_NCF_predictedSubmission.csv', all_predictions.cpu().detach().numpy())
