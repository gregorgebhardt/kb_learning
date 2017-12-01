#!/bin/bash
#SBATCH -A project00554
#SBATCH -J pendulum_filtering
#SBATCH -D /home/yy05vipo/git/pyKKF/experiments
#SBATCH --mail-type=END
# Please use the complete path details :
#SBATCH -e /home/yy05vipo/git/pyKKF/experiments/filtering/pendulum/err.%j
#SBATCH -o /home/yy05vipo/git/pyKKF/experiments/filtering/pendulum/out.%j
#
#SBATCH -n 40      # Number of tasks
#SBATCH -c 8       # Number of cores per task
#SBATCH --mem-per-cpu=2000  # Main memory in MByte per MPI task
#SBATCH -t 2:00:00     # Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes

# -------------------------------
# Afterwards you write your own commands, e.g.
module load gcc openmpi/gcc python/3.6 intel
source /home/yy05vipo/.virtenvs/kkf/bin/activate
cd /home/yy05vipo/git/pyKKF/experiments
srun hostname > hostfile.$SLURM_JOB_ID
hostfileconv hostfile.$SLURM_JOB_ID
job_stream --hostfile hostfile.$SLURM_JOB_ID.converted -- python filtering/pendulum/experiment.py -c filtering/pendulum/experiment.yml -v