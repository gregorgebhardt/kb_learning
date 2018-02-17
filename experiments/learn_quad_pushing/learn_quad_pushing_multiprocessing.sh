#!/bin/bash
#SBATCH -A project00672 # 672
#SBATCH -J learn_quad_pushing
#SBATCH -D /home/yy05vipo/ftp/kb_learning/experiments
#SBATCH --mail-type=END
# Please use the complete path details :
#SBATCH -e /home/yy05vipo/ftp/kb_learning/experiments/learn_quad_pushing/l_%j.stderr
#SBATCH -o /home/yy05vipo/ftp/kb_learning/experiments/learn_quad_pushing/l_%j.stdout
#
#SBATCH -n 50               # Number of tasks
#SBATCH -c 8                # Number of cores per task
#SBATCH --mem-per-cpu=2000  # Main memory in MByte per MPI task
#SBATCH -t 12:00:00         # Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes
### SBATCH --hint=multithread

# -------------------------------
# Afterwards you write your own commands, e.g.

#module purge
#module load gcc/4.9.4 openmpi/gcc/2.1.2 python/3.6.2 intel/2018u1 boost/1.61

source /home/yy05vipo/.virtenvs/gym/bin/activate
cd /home/yy05vipo/ftp/kb_learning/experiments

srun hostname > $SLURM_JOB_ID.hostfile
hostfileconv $SLURM_JOB_ID.hostfile -1
job_stream --hostfile $SLURM_JOB_ID.hostfile.converted -- python learn_quad_pushing/learn_quad_pushing.py -c learn_quad_pushing/learn_quad_pushing_multiprocessing.yml --log_level DEBUG