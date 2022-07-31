# Alternating least squares ensemble (ALS ens.)

### Model description


### Dependencies
LibRecommender library: https://github.com/massquantity/LibRecommender
```
pip install LibRecommender==0.8.4
```

### Files
 - `run_prepare_ensemble_final.py`: Computes the final prediction on the test dataset, using all available train data.
 - `run_prepare_ensemble_validation.py`: Requires as first argument the split id(between 0 and 4). It trains the model on `data_val_train_kfold/partition_k_train.csv` and evaluates on the user/movie pair specified by `data_val_train_kfold/partition_k_val.csv`.
 - `dataset.py`: helper functions to load and save the dataset