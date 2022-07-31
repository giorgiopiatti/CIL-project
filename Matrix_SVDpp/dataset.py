import numpy as np
import pandas as pd

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

    return users, movies, predictions


def save_predictions(res_path, predictions):
    test_pd = pd.read_csv(DATA_DIR+'/sampleSubmission.csv')

    test_pd = test_pd.astype({'Prediction': 'float'})

    test_pd.iloc[:, 1] = predictions

    test_pd.to_csv(res_path, index=False, float_format='%.3f')


def save_predictions_from_pandas(res_path, predictions, index_pd):
    index_pd = index_pd.astype({'Prediction': 'float'})
    index_pd.iloc[:, 1] = predictions
    index_pd.to_csv(res_path, index=False, float_format='%.3f')
