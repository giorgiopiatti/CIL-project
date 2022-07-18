import numpy as np 
import torch 
import pandas as pd
from scipy.sparse import coo_matrix


number_of_users, number_of_movies = (10000, 1000)
DATA_DIR = './data'

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


class MaskedUserDatasetTrain(torch.utils.data.Dataset):
    def __init__(self,matrix_users_movies, user_movies_idx, num_batch_samples,
    num_masked_movies):
        self.matrix= matrix_users_movies
        self.idx = user_movies_idx

        self.num_batch_samples = num_batch_samples
        self.num_masked_movies = num_masked_movies
        

    def __getitem__(self, index):

        index = index % number_of_users
        user = torch.tensor(self.matrix[index])
        user_true = user.clone().detach()
        
        l = len(self.idx[index])
        if l >= 2:
            k = min(l-1, self.num_masked_movies)
            
            k = torch.randint(low=1,high=k+1, size=(1,))

            perm = torch.randperm(l).cpu()
            idx = perm[:k]

            mask = torch.zeros_like(user, dtype=torch.bool)
            for a in idx:
                id = self.idx[index][a]
                mask[id] = True
                user[id] = 0.

            return user, mask, user_true

    def __len__(self):
        return number_of_users*self.num_batch_samples


class MaskedUserDatasetVal(torch.utils.data.Dataset):
    def __init__(self,matrix_train, matrix_val):
        super().__init__()
        self.matrix_train = matrix_train
        self.matrix_val = matrix_val
    
    def __getitem__(self, index):
        user_train = torch.tensor(self.matrix_train[index])
        user_val = torch.tensor(self.matrix_val[index])
        
        mask = user_val.type(torch.bool)
        return user_train, mask, user_val

    
    def __len__(self):
        return number_of_users


class DatasetComplete(torch.utils.data.Dataset):
    def __init__(self,matrix):
        super().__init__()
        self.matrix = matrix
    
    
    def __getitem__(self, index):
        user = torch.tensor(self.matrix[index])
        return user
    
    def __len__(self):
        return len(self.matrix)



