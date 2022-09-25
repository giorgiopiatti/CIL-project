# Collaborative models for Collaborative Filtering
In the age of information, users are feeling overwhelmed by countless offerings and services. Recommender systems aim to alleviate this issue by presenting personalized recommendations to the user. They are often based on Collaborative filtering (CF), which analyzes past transactions of the user in order to find new recommendations. 

We propose a new model that combines the existing results of many years of research on CF. In total, we adopt four different types of approaches: RMF, Autoencoder, SVD++ and NCF. For some approaches, instead of optimizing parameters, we combine the results using an ensemble. \\In addition, we introduce a probabilistic approach for Neural Collaborative filtering, which drastically outperforms current NCF models. 

#### Authors
Nathan Corecco, Giorgio Piatti, David Gu


## Overview

```
/AE                         -- Autoencoder
/ALS                        -- Alternating least squares (baseline)
/ALS_ensemble               -- Alternating least squares ensemble
/data                       -- Kaggle dataset
/data_val_train_kfold       -- 5 partition to compute CV score
/NCF                        -- Neural Collaborative Filtering (baseline)
/PNCF                       -- Probabilistic Neural Collaborative Filtering
/PNCF_base                  -- Probabilistic Neural Collaborative Filtering, base model (without FGE)
/results_baseline           -- Results of baseline models (need to download, see additional data section)
/results_ensemble           -- Results of sub model of the ensemble (need to download, see additional data section)
/SVDpp                      -- SVD++ (baseline)
/SVDpp_ensemble             -- SVD++ ensemble
analysis_baseline.ipynb     -- Compute score on CV split for baseline models
analysis_ensemble.ipynb     -- Compute score on CV split for the final ensemble
ensemble_lsq.ipynb          -- Compute final result of the ensemble
requirements.txt            -- Requirement file of the necessary libraries
/paper_generate_graphs      -- Code used to generate the graphs in the paper
paper.pdf                   -- paper describing our approaches
```

Each baseline model folder (ALS, NCF, SVDpp) shares the same general structure, namely
 - `run_prepare_baseline_final.py`: Computes the final prediction on the test dataset, using all available train data.
 - `run_prepare_baseline_validation.py`: Requires as first argument the split id(between 0 and 4). It trains the model on `data_val_train_kfold/partition_k_train.csv` and evaluates on the user/movie pair specified by `data_val_train_kfold/partition_k_val.csv`.
Those 2 files are used to computed the predictions for model evaluation, which are saved for convenience under the folder `results_baseline`.

Each sub model folder (AE, ALS_ensemble, PNCF, SVDpp_ensemble) shares the same general structure, namely
- `run_prepare_ensemble_final.py`: Computes the final prediction on the test dataset, using all available train data.
- `run_prepare_ensemble_validation.py`: Requires as first argument the split id(between 0 and 4). It trains the model on `data_val_train_kfold/partition_k_train.csv` and evaluates on the user/movie pair specified by `data_val_train_kfold/partition_k_val.csv`.
Those 2 files are used to computed the predictions for model evaluation and for computing the weightning coefficient for the final ensemble, which are saved for convenience under the folder `results_ensemble`.

#### Additional data
We already provided the intermediate prediciton to be able to reproduce the final ensemble result.
- Download the folder https://drive.infomaniak.com/app/share/105096/7726440d-25c5-44d3-b586-3a3dd69d30b0 and unzip it as `results_ensemble` in the main directory
- Download the folder https://drive.infomaniak.com/app/share/105096/647f97fd-a3f9-4375-afc8-c49a09379579 and unzip it as `results_baseline` in the main directory

### Submission generation
- To reproduce the intermediate results (i.e. to compute the content of the folder `results_ensemble`) run the sh script `prepare_ensemble.sh` or on Euler `euler_prepare_ensemble.sh` from the main folder. Note on Euler it's best to modify the env variable `BASE_DIR_RESULTS` defined in the `.env` file to point to a scratch directory.

- Run the notebook `ensemble_lsq.ipynb`, the results are stored in `AE-ALS_ensemble-PNCF-SVDpp_ensemble-LSQ-results.csv`

### Baseline score evaluation
- To reproduce the intermediate results (i.e. to compute the content of the folder `results_baseline`) run the sh script `prepare_baseline.sh` or on Euler `euler_prepare_baseline.sh` from the main folder. Note on Euler it's best to modify the env variable `BASE_DIR_RESULTS` defined in the `.env` file to point to a scratch directory.

- Run the notebook `ensemble_lsq.ipynb`, the results are stored in `AE-ALS_ensemble-PNCF-SVDpp_ensemble-LSQ-results.csv`


### Installation
To run the code, it is recommended to first run the following instructions:
`pip install -r requirements.txt`
