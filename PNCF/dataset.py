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
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract(
            'r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values

    return users, movies, predictions-1


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
        self.ratings = ratings
        self.is_test_dataset = is_test_dataset

    def __getitem__(self, index):
        user = self.users[index]
        movie = self.movies[index]
        rating = self.ratings[index]

        x = torch.tensor([user, movie])
        y = torch.tensor(rating)
        if self.is_test_dataset:
            return x
        return x, y

    def __len__(self):
        return len(self.users)


def save_predictions(res_path, predictions):
    test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')

    test_pd = test_pd.astype({'Prediction': 'float'})

    test_pd.iloc[:, 1] = predictions

    test_pd.to_csv(res_path, index=False, float_format='%.3f')


def save_predictions_from_pandas(res_path, predictions, index_pd):
    index_pd = index_pd.astype({'Prediction': 'float'})
    index_pd.iloc[:, 1] = predictions
    index_pd.to_csv(res_path, index=False, float_format='%.3f')
