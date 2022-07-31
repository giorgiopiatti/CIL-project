for i in `seq 0 4`
do
    bsub -n 16 -W 1:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python ./ALS/run_prepare_baseline_validation.py $i
done

bsub -n 16 -W 1:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python ./ALS/run_prepare_baseline_final.py


for i in `seq 0 4`
do
    bsub -n 16 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python ./NCF/run_prepare_baseline_validation.py $i
done

bsub -n 16 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python ./NCF/run_prepare_baseline_final.py
