qsub -q gpu -l select=1:ncpus=1:ngpus=1:gpu_cap=cuda35:mem=64gb:scratch_local=100gb -l walltime=05:00:00 job.sh >> runnin_jobs
