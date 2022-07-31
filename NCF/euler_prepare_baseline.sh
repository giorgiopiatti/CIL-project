for i in `seq 0 4`
do
    bsub -n 16 -W 4:00 -R "rusage[mem=2048]" python run_prepare_baseline_validation.py $i
done

bsub -n 16 -W 4:00 -R "rusage[mem=2048]" python run_prepare_baseline_final.py