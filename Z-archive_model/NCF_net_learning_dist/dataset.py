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

    return users, movies, predictions-1
    

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

class TripletDataset(torch.utils.data.Dataset):
    """
    Dataset
    x = (movie, user)
    y = rating

    """
    def __init__(self, users, movies, ratings, is_test_dataset=False) -> None:
        super().__init__()
        self.users = users
        self.movies = movies
        self.ratings =ratings
        self.is_test_dataset = is_test_dataset

    
    def __getitem__(self, index):
        user = self.users[index]
        movie = self.movies[index]
        rating = self.ratings[index]

        x = torch.tensor([user, movie])
        y = torch.tensor(rating)
        if self.is_test_dataset:
            return x
        return x,y
    
    def __len__(self):
        return len(self.users)


def save_predictions(res_path, predictions):
    test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')
   
    test_pd = test_pd.astype({'Prediction': 'float'})

    test_pd.iloc[:, 1] = predictions

 
    test_pd.to_csv(res_path, index=False, float_format='%.3f')
