for i in `seq 0 4`
do
    bsub -n 16 -W 1:00 -R "rusage[mem=2048]" python run_prepare_ensemble_validation.py $i
done

bsub -n 16 -W 1:00 -R "rusage[mem=2048]" python run_prepare_ensemble_final.py