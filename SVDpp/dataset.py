import numpy as np 
import torch 
import pandas as pd
from scipy.sparse import coo_matrix


number_of_users, number_of_movies = (10000, 1000)
DATA_DIR = '../data'

def extract_users_movies_ratings_lists(data_pd):
    """
    Return 3 lists containing user, movies, and ratins
    """
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values

    return users, movies, predictions
    

def extract_matrix_users_movies_ratings(data_pd):
    """
    Returns:
    - sparse matrix filled with ratings
    - list of non-zero entries indexed by user
    """
    users, movies, predictions = extract_users_movies_ratings_lists(data_pd)

    matrix_users_movies = coo_matrix((predictions, (users, movies)), dtype=np.float32 ).toarray()
    user_movies_idx = []
    for _ in range(number_of_users):
        user_movies_idx.append([])

    nnz = np.nonzero(matrix_users_movies)
    for n in range(len(nnz[0])):
        user_movies_idx[nnz[0][n]].append(nnz[1][n])

    return matrix_users_movies, user_movies_idx

def save_predictions(res_path, reconstructed_matrix):

    test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
  
    users, movies = \
        [np.squeeze(arr) for arr in np.split(test_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
   
    predictions = np.zeros(len(users))

    for i, (user, movie) in enumerate(zip(users, movies)):
        predictions[i] = reconstructed_matrix[user][movie]

    test_pd = test_pd.astype({'Prediction': 'float'})
    test_pd.iloc[:, 1] = predictions

    test_pd.to_csv(res_path, index=False, float_format='%.3f')



def save_predictions_from_pandas(res_path, reconstructed_matrix, index_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(index_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
   
    predictions = np.zeros(len(users))

    for i, (user, movie) in enumerate(zip(users, movies)):
        predictions[i] = reconstructed_matrix[user][movie]

    index_pd = index_pd.astype({'Prediction': 'float'})
    index_pd.iloc[:, 1] = predictions

    index_pd.to_csv(res_path, index=False, float_format='%.3f')
