for i in `seq 0 4`
do
    bsub -n 16 -W 1:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python ./AE/run_prepare_ensemble_validation.py $i
done

bsub -n 16 -W 1:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python ./AE/run_prepare_ensemble_final.py

for i in `seq 0 4`
do
    bsub -n 16 -W 4:00 -J "job$i" -R "rusage[mem=2048, ngpus_excl_p=1]" python ./PNCF_base/run_prepare_ensemble_validation.py $i
    bsub -n 16 -W 4:00 -w "done(job$i)" -R "rusage[mem=2048, ngpus_excl_p=1]" python ./PNCF/run_prepare_ensemble_validation.py $i
done

bsub -n 16 -W 4:00 -J "final" -R "rusage[mem=2048, ngpus_excl_p=1]" python ./PNCF_base/run_prepare_ensemble_final.py
bsub -n 16 -W 4:00 -w "done(final)" -R "rusage[mem=2048, ngpus_excl_p=1]" python ./PNCF/run_prepare_ensemble_final.py


for i in `seq 0 4`
do
    bsub -n 16 -W 4:00 -R "rusage[mem=2048]" python ./ALS_ensemble/run_prepare_ensemble_validation.py $i
done

bsub -n 16 -W 4:00 -R "rusage[mem=2048]" python ./ALS_ensemble/run_prepare_ensemble_final.py



for i in `seq 0 4`
do
    bsub -n 16 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python ./SVDpp_ensemble/run_prepare_ensemble_validation.py $i
done

bsub -n 16 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python ./SVDpp_ensemble/run_prepare_ensemble_final.py
