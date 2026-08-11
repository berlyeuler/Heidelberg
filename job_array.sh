#!/bin/bash

# Standard output and error format (Hocanın istediği gibi):
#SBATCH -o job.%A_%a.out
#SBATCH -e job.%A_%a.err

# Initial working directory:
#SBATCH -D ./

# Partition to use:
#SBATCH -p normal

# Job Name and details:
#SBATCH --job-name=movie_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=01:30:00

# 10 adet simülasyon için Array tanımı:
#SBATCH --array=8


module load Python/3.11.5 

dirname_list=(sim_0 sim_1 sim_2 sim_3 sim_4 sim_5 sim_6 sim_7 sim_8 sim_9)


dirname=${dirname_list[$SLURM_ARRAY_TASK_ID]}

echo "simulation: $dirname"


srun python ${HOME}/repos/Heidelberg/slurm_movie.py "$dirname"


TARGET_DIR="/scratch/hpc-prf-radmix/hpcbeoe/${dirname}"

mv job.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ${TARGET_DIR}/
mv job.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err ${TARGET_DIR}/