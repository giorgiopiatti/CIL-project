for i in `seq 0 4`
do
    python ./ALS/run_prepare_baseline_validation.py $i
done

python ./ALS/run_prepare_baseline_final.py


for i in `seq 0 4`
do
    python ./NCF/run_prepare_baseline_validation.py $i
done

python ./NCF/run_prepare_baseline_final.py
