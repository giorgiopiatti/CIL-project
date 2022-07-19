for i in `seq 0 4`
do
    bsub -n 16 -W 1:00 -G ls_lawecon -R "rusage[mem=2048, ngpus_excl_p=1]" python run_prepare_ensemble_validation.py $i
done

bsub -n 16 -W 1:00 -G ls_lawecon -R "rusage[mem=2048, ngpus_excl_p=1]" python run_prepare_ensemble_final.py