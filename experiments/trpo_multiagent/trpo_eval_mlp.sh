#!/bin/bash
#SBATCH -A project00720
#SBATCH -J trpo_ma_mlp
#SBATCH -a 0-4
#SBATCH -D /home/yy05vipo/git/kb_learning/experiments
#SBATCH --mail-type=END
# Please use the complete path details :
#SBATCH -e /home/yy05vipo/git/kb_learning/experiments/trpo_multiagent/l_%A_%a.stderr
#SBATCH -o /home/yy05vipo/git/kb_learning/experiments/trpo_multiagent/l_%A_%a.stdout
#
#SBATCH -n 20               # Number of tasks
#SBATCH -c 1                # Number of cores per task
#SBATCH --mem-per-cpu=2000  # Main memory in MByte per MPI task
#SBATCH -t 5:00:00          # Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes
#SBATCH -C avx2             # requires new nodes
### SBATCH --hint=multithread

# -------------------------------
# Afterwards you write your own commands, e.g.

module purge
module load gcc/4.9.4 openmpi/gcc

. /home/yy05vipo/bin/miniconda3/etc/profile.d/conda.sh
conda activate dme_wo_mpi

cd /home/yy05vipo/git/kb_learning/experiments

mpiexec -np 20 -map-by core -bind-to core python trpo_multiagent/trpo_ma.py trpo_multiagent/trpo_ma.yml \
    -me eval_mlp -g 1 -l DEBUG -j $SLURM_ARRAY_TASK_ID
