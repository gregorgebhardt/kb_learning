#!/bin/bash
#SBATCH -A project00720
#SBATCH -J ppo_test
#SBATCH -D /home/yy05vipo/git/kb_learning/experiments
#SBATCH --mail-type=END
# Please use the complete path details :
#SBATCH -e /home/yy05vipo/git/kb_learning/experiments/ppo/l_%j.stderr
#SBATCH -o /home/yy05vipo/git/kb_learning/experiments/ppo/l_%j.stdout
#
#SBATCH -n 6                # Number of tasks
#SBATCH -c 8                # Number of cores per task
#SBATCH --mem-per-cpu=1000   # Main memory in MByte per MPI task
#SBATCH -t 30         # Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes
#SBATCH -C avx2            # requires new nodes
### SBATCH --hint=multithread

# -------------------------------
# Afterwards you write your own commands, e.g.

module purge
module load gcc

#. /home/yy05vipo/bin/miniconda3/etc/profile.d/conda.sh
#conda activate dme
#
#cd /home/yy05vipo/git/kb_learning/experiments

# we want to run one instance in addition on the main node
hostname > $SLURM_JOB_ID.hostfile

mpirun hostname >> $SLURM_JOB_ID.hostfile

#HYDRA_TOPO_DEBUG=1 mpiexec -hostfile $SLURM_JOB_ID.hostfile python ppo/ppo.py ppo/ppo.yml -me test -l DEBUG
#
#rm $SLURM_JOB_ID.hostfile
