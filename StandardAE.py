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
    

number_of_users, number_of_movies = (10000, 1000)
rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))

from sklearn.metrics import mean_squared_error

rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))

def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

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
    test_pd = pd.read_csv('../data/sampleSubmission.csv')
    test_pd = test_pd.astype({'Prediction': 'float'})
    test_pd.iloc[:, 1] = predictions
    test_pd.to_csv(res_path, index=False, float_format='%.3f')

df_trains = []
df_vals = []

for i in range(5):
    df_trains.append(pd.read_csv(f'data_val_train_kfold/partition_{i}_train.csv'))
    df_vals.append(pd.read_csv(f'data_val_train_kfold/partition_{i}_val.csv'))

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

class Encoder(nn.Module):
    def __init__(self, input_dimension, encoded_dimension=16):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dimension, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=encoded_dimension),
            nn.ReLU()
        )
    
    def forward(self, data):
        return self.model(data)

class Decoder(nn.Module):
    def __init__(self, output_dimensions, encoded_dimension=16):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=encoded_dimension, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dimensions),
            nn.ReLU() # How does the output look like? What about if you had first centered the data?!
        )
    
    def forward(self, encodings):
        return self.model(encodings)


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(self.encoder(data))

# Parameters
batch_size = 64
num_epochs = 1000
show_validation_score_every_epochs = 5
encoded_dimension = 16
learning_rate = 1e-3

# Model

# L2 loss between original ratings and reconstructed ratings for the observed values
def loss_function(original, reconstructed, mask):
    return torch.mean(mask * (original - reconstructed) ** 2)

# reconstuct the whole array
def reconstruct_whole_matrix(autoencoder):
    data_reconstructed = np.zeros((number_of_users, number_of_movies))
    
    with torch.no_grad():
        for i in range(0, number_of_users, batch_size):
            upper_bound = min(i + batch_size, number_of_users)
            data_reconstructed[i:upper_bound] = autoencoder(data_torch[i:upper_bound]).detach().cpu().numpy()

    return data_reconstructed


def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2)


scores = []

for i in range(5):
    train_pd = df_trains[i]
    val_pd = df_vals[i]
    train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)

    # also create full matrix of observed values
    data = np.full((number_of_users, number_of_movies), np.mean(train_pd.Prediction.values))
    mask = np.zeros((number_of_users, number_of_movies)) # 0 -> unobserved value, 1->observed value


    train_users, train_movies, train_predictions = extract_users_movies_ratings_lists(train_pd)
    test_users, test_movies, test_predictions = extract_users_movies_ratings_lists(val_pd)

    def extract_prediction_from_full_matrix(reconstructed_matrix, users=test_users, movies=test_movies):
        # returns predictions for the users-movies combinations specified based on a full m \times n matrix
        assert(len(users) == len(movies)), "users-movies combinations specified should have equal length"
        predictions = np.zeros(len(test_users))

        for i, (user, movie) in enumerate(zip(users, movies)):
            predictions[i] = reconstructed_matrix[user][movie]

        return predictions

    for user, movie, pred in zip(train_users, train_movies, train_predictions):
        data[user - 1][movie - 1] = pred
        mask[user - 1][movie - 1] = 1

    autoencoder = AutoEncoder(
        encoder=Encoder(
            input_dimension=number_of_movies,
            encoded_dimension=encoded_dimension,
        ),
        decoder=Decoder(
            output_dimensions=number_of_movies,
            encoded_dimension=encoded_dimension,
        )
    ).to(device)

    optimizer = optim.Adam(autoencoder.parameters(),
                        lr=learning_rate)

    # Build Dataloaders
    data_torch = torch.tensor(data, device=device).float()
    mask_torch = torch.tensor(mask, device=device)

    dataloader = DataLoader(
        TensorDataset(data_torch, mask_torch),
        batch_size=batch_size)


    # TRAINING
    step = 0
    with tqdm(total=len(dataloader) * num_epochs) as pbar:
        for epoch in range(num_epochs):
            for data_batch, mask_batch in dataloader:
                optimizer.zero_grad()

                reconstructed_batch = autoencoder(data_batch)

                loss = loss_function(data_batch, reconstructed_batch, mask_batch)

                loss.backward()

                optimizer.step()

                pbar.update(1)
                step += 1

    # EVALUATE
    reconstructed_matrix = reconstruct_whole_matrix(autoencoder)
    predictions = extract_prediction_from_full_matrix(reconstructed_matrix)
    score= rmse(predictions, test_predictions)

    print(score)
    scores.append(score)

print(np.array(scores).mean())
print(np.array(scores).std())

np.save('scoresNCF', scores)
