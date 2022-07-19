for i in `seq 0 4`
do
    bsub -n 16 -W 4:00 -G ls_lawecon -R "rusage[mem=2048, ngpus_excl_p=1]" python SVDpp_ensemble_gaussian_prepare_ensemble_validation.py $i
done

bsub -n 16 -W 4:00 -G ls_lawecon -R "rusage[mem=2048, ngpus_excl_p=1]" python SVDpp_ensemble_gaussian_prepare_ensemble_final.py