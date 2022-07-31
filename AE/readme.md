# Autoencoder (AE)
### Model description


### Files
 - `model.py` : Autoencoder model
 - `swa_model.py`: Wrapper to perform Fast Geometric Ensembling of the model specified by `model.py`
 - `run_base.py`: Train AE base model
 - `run_swa.py`: Train AE base model and perform Fast Geometric Ensembling
 - `run_optuna_swa.py`: Perform Bayesian Optimizazion to find the hyperparameters of Fast Geometric Ensembling, using the library Optuna. Need to generate base model checkpoint via `run_base.py`
 - `run_optuna.py`: Perform Bayesian Optimizazion to find the hyperparameters of the base model, using the library Optuna.

 - `run_prepare_ensemble_final.py`: Computes the final prediction on the test dataset, using all available train data.
 - `run_prepare_ensemble_validation.py`: Requires as first argument the split id(between 0 and 4). It trains the model on `data_val_train_kfold/partition_k_train.csv` and evaluates on the user/movie pair specified by `data_val_train_kfold/partition_k_val.csv`.

 - `dataset.py`: helper functions to load and save the dataset
 - `ex_dask.py`, `optuna_run_dask.py`, `optuna_single_gpu.py`: helper files to perform parallel Bayesian Optimization via Optuna