import optuna
import joblib

TIMEOUT_OPTUNA = 22*60*60 # 22h
DIR_RESULTS = '/cluster/scratch/piattigi/CIL/res_optuna'


from dask.distributed import Client
from ex_dask import create_study, DaskStorage

"""
trial_fn(trial: Trial, experiment_name: string, gpu_id: int)

"""
def run_optuna(trial_fn, experiment_name, n_trials, n_gpus):
    # Everything should be inside client scope, otherwise we cannot access storage, 
    # since it's all dask distributed based. 
    #
    # Cannot log to Neptute logger: some pickle error
   
    from dask_cuda import LocalCUDACluster
    with LocalCUDACluster(n_workers=n_gpus) as cluster, Client(cluster) as client:
        storage = DaskStorage(client=client)
        study = create_study(direction='minimize', study_name=experiment_name, storage=storage)

        study.optimize(trial_fn, n_trials=n_trials, timeout=TIMEOUT_OPTUNA, client=client)
        
        joblib.dump(study, f"{DIR_RESULTS}/{experiment_name}/{experiment_name}-study.pkl")
        print("[OPTUNA]  Best score: {}".format(study.best_value))
        print("[OPTUNA]  Best params: {}".format(study.best_params))

        study.trials_dataframe().to_csv(f'{DIR_RESULTS}/{experiment_name}/{experiment_name}-optuna_results.csv')
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(f'{DIR_RESULTS}/{experiment_name}/{experiment_name}-param_importances.png')
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(f'{DIR_RESULTS}/{experiment_name}/{experiment_name}-optimization_history.png')
        except:
            pass
        
        return study.best_params