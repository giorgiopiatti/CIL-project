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

    # choose how many singlular values to keep
    k_singular_values = 2
    number_of_singular_values = min(number_of_users, number_of_movies)

    assert(k_singular_values <= number_of_singular_values), "choose correct number of singular values"

    U, s, Vt = np.linalg.svd(data, full_matrices=False)

    S = np.zeros((number_of_movies, number_of_movies))
    S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])

    reconstructed_matrix = U.dot(S).dot(Vt)
        
    predictions = extract_prediction_from_full_matrix(reconstructed_matrix)

    # EVALUATE
    score= rmse(predictions, test_predictions)

    print(score)
    scores.append(score)

print(np.array(scores).mean())
print(np.array(scores).std())

np.save('StandardSVD', scores)

