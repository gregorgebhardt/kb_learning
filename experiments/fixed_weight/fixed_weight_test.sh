#!/bin/bash
#SBATCH -A project00720
#SBATCH -J fw_test
#SBATCH -D /home/yy05vipo/git/kb_learning/experiments
#SBATCH --mail-type=END
# Please use the complete path details :
#SBATCH -e /home/yy05vipo/git/kb_learning/experiments/fixed_weight/l_%j.stderr
#SBATCH -o /home/yy05vipo/git/kb_learning/experiments/fixed_weight/l_%j.stdout
#
#SBATCH -n 4               # Number of tasks
#SBATCH -c 6                # Number of cores per task
#SBATCH --mem-per-cpu=1000   # Main memory in MByte per MPI task
#SBATCH -t 2:00:00         # Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes
#SBATCH -C avx2            # requires new nodes
### SBATCH --hint=multithread

# -------------------------------
# Afterwards you write your own commands, e.g.

module purge
module load gcc openmpi/gcc/2.1

export OMP_NUM_THREADS=8

. /home/yy05vipo/bin/miniconda3/etc/profile.d/conda.sh
conda activate dme

cd /home/yy05vipo/git/kb_learning/experiments

srun hostname > $SLURM_JOB_ID.hostfile
#hostfileconv $SLURM_JOB_ID.hostfile -1

#job_stream --hostfile $SLURM_JOB_ID.hostfile.converted -- python fixed_weight/fixed_weight.py -c fixed_weight/fixed_weight.yml -e test -d -l DEBUG
#mpirun -hostfile $SLURM_JOB_ID.hostfile -np 1 --mca mpi_warn_on_fork 0 --display-allocation --display-map python fixed_weight/fixed_weight.py fixed_weight/fixed_weight.yml -e test -c -l DEBUG
mpiexec -map-by node -hostfile $SLURM_JOB_ID.hostfile --mca mpi_warn_on_fork 0 --display-allocation --display-map python -m mpi4py fixed_weight/fixed_weight.py fixed_weight/fixed_weight.yml -e test -c -o -l DEBUG


rm $SLURM_JOB_ID.hostfile
#rm $SLURM_JOB_ID.hostfile.converted
