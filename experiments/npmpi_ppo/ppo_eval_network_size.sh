#!/bin/bash
#SBATCH -A project00720
#SBATCH -J ppo_eval_network_size
#SBATCH -D /home/yy05vipo/git/kb_learning/experiments
#SBATCH --mail-type=END
# Please use the complete path details :
#SBATCH -e /home/yy05vipo/git/kb_learning/experiments/npmpi_ppo/l_%j.stderr
#SBATCH -o /home/yy05vipo/git/kb_learning/experiments/npmpi_ppo/l_%j.stdout
#
#SBATCH -n 24               # Number of tasks
#SBATCH -c 6                # Number of cores per task
#SBATCH --mem-per-cpu=1000   # Main memory in MByte per MPI task
#SBATCH -t 3:00:00         # Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes
#SBATCH -C avx2            # requires new nodes
### SBATCH --hint=multithread

# -------------------------------
# Afterwards you write your own commands, e.g.

module purge
module load gcc/4.9.4 openmpi/gcc

. /home/yy05vipo/bin/miniconda3/etc/profile.d/conda.sh
conda activate dme_wo_mpi

cd /home/yy05vipo/git/kb_learning/experiments

mpiexec -map-by node python ppo/ppo.py ppo/ppo.yml -dme eval_network_size -l DEBUG
