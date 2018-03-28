#!/bin/bash
#SBATCH -A project00672 # 672, 664
#SBATCH -J bandwidth_eval
#SBATCH -D /home/yy05vipo/git/kb_learning/experiments
#SBATCH --mail-type=END
# Please use the complete path details :
#SBATCH -e /home/yy05vipo/git/kb_learning/experiments/fixed_weight/l_%j.stderr
#SBATCH -o /home/yy05vipo/git/kb_learning/experiments/fixed_weight/l_%j.stdout
#
#SBATCH -n 12               # Number of tasks
#SBATCH -c 8                # Number of cores per task
#SBATCH --mem-per-cpu=1000   # Main memory in MByte per MPI task
#SBATCH -t 6:00:00         # Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes
#SBATCH -C avx2            # requires new nodes
### SBATCH --hint=multithread

# -------------------------------
# Afterwards you write your own commands, e.g.

#module purge
#module load gcc/4.9.4 openmpi/gcc/2.1.2 python/3.6.2 intel/2018u1 boost/1.61

source /home/yy05vipo/.virtenvs/gym/bin/activate
cd /home/yy05vipo/git/kb_learning/experiments

srun hostname > $SLURM_JOB_ID.hostfile
hostfileconv $SLURM_JOB_ID.hostfile -1

job_stream --hostfile $SLURM_JOB_ID.hostfile.converted -- python fixed_weight/fixed_weight.py -c fixed_weight/fixed_weight.yml --log_level INFO -e bandwidth_eval -o

rm $SLURM_JOB_ID.hostfile
rm $SLURM_JOB_ID.hostfile.converted
