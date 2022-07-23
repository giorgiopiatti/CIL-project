import optuna
import joblib

TIMEOUT_OPTUNA = 22*60*60 # 22h
DIR_RESULTS = '/cluster/scratch/piattigi/CIL/res_optuna'

from optuna import create_study
"""
trial_fn(trial: Trial, experiment_name: string, gpu_id: int)

"""
def run_optuna(trial_fn, experiment_name, n_trials):
    study = create_study(direction='minimize', study_name=experiment_name)

    study.optimize(trial_fn, n_trials=n_trials, timeout=TIMEOUT_OPTUNA)
    
    joblib.dump(study, f"{DIR_RESULTS}/{experiment_name}/{experiment_name}-study.pkl")
    print("[OPTUNA]  Best score: {}".format(study.best_value))
    print("[OPTUNA]  Best params: {}".format(study.best_params))

    return study.best_params