# Alternating least squares (ALS)

### Model description
ALS is a classical technique used to solve the classical regularized matrix factorization problem. The model tries to find a low rank approximation for the rating matrix $Y$ by finding two matrices $U,M$ such that $Y \approx UM$. ALS, as the name suggests, approaches the problem by optimizing alternately $U$ and $M$.

### Dependencies
LibRecommender library: https://github.com/massquantity/LibRecommender
```
pip install LibRecommender==0.8.4
```

### Files
 - `run_prepare_baseline_final.py`: Computes the final prediction on the test dataset, using all available train data.
 - `run_prepare_baseline_validation.py`: Requires as first argument the split id(between 0 and 4). It trains the model on `data_val_train_kfold/partition_k_train.csv` and evaluates on the user/movie pair specified by `data_val_train_kfold/partition_k_val.csv`.
 - `dataset.py`: helper functions to load and save the dataset