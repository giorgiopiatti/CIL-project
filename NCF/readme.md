# Neural Collaborative Filtering (NCF) 

### Model description
Classical neural collaborative filtering, neural network model, which tries to learn users and movies embeddings and also the funtion used to combine them in order to get the score. Trained with a classical RMSE.

### Files
 - `run_prepare_baseline_final.py`: Computes the final prediction on the test dataset, using all available train data.
 - `run_prepare_baseline_validation.py`: Requires as first argument the split id(between 0 and 4). It trains the model on `data_val_train_kfold/partition_k_train.csv` and evaluates on the user/movie pair specified by `data_val_train_kfold/partition_k_val.csv`.
 - `dataset.py`: helper functions to load and save the dataset