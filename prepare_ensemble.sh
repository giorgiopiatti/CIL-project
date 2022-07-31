for i in `seq 0 4`
do
    python ./AE/run_prepare_ensemble_validation.py $i
done

python ./AE/run_prepare_ensemble_final.py

for i in `seq 0 4`
do
    python ./PNCF_base/run_prepare_ensemble_validation.py $i
    python ./PNCF/run_prepare_ensemble_validation.py $i
done

python ./PNCF_base/run_prepare_ensemble_final.py
python ./PNCF/run_prepare_ensemble_final.py


for i in `seq 0 4`
do
    python ./ALS_ensemble/run_prepare_ensemble_validation.py $i
done

python ./ALS_ensemble/run_prepare_ensemble_final.py


for i in `seq 0 4`
do
    python ./SVDpp_ensemble/run_prepare_ensemble_validation.py $i
done
python ./SVDpp_ensemble/run_prepare_ensemble_final.py