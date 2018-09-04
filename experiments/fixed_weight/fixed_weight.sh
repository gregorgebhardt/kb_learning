#!/bin/bash
#SBATCH -A project00672
#SBATCH -J fw_square
#SBATCH -D /home/yy05vipo/git/kb_learning/experiments
#SBATCH --mail-type=END
# Please use the complete path details :
#SBATCH -e /home/yy05vipo/git/kb_learning/experiments/fixed_weight/l_%j.stderr
#SBATCH -o /home/yy05vipo/git/kb_learning/experiments/fixed_weight/l_%j.stdout
#
#SBATCH -n 4               # Number of tasks
#SBATCH -c 6                # Number of cores per task
#SBATCH --mem-per-cpu=1000   # Main memory in MByte per MPI task
#SBATCH -t 5:00:00         # Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes
#SBATCH -C avx2            # requires new nodes
### SBATCH --hint=multithread

# -------------------------------
# Afterwards you write your own commands, e.g.

#module purge
module load git gcc openmpi/intel/2.1.2 intel

export LD_LIBRARY_PATH="/home/yy05vipo/lib/boost_1_61_0/stage/lib:$LD_LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="/home/yy05vipo/lib/boost_1_61_0:$CPLUS_INCLUDE_PATH"

source /home/yy05vipo/.virtenvs/gym/bin/activate
cd /home/yy05vipo/git/kb_learning/experiments

srun hostname > $SLURM_JOB_ID.hostfile
hostfileconv $SLURM_JOB_ID.hostfile -1

job_stream --hostfile $SLURM_JOB_ID.hostfile.converted -- python fixed_weight/fixed_weight.py -c fixed_weight/fixed_weight.yml --log_level DEBUG -e square -o

rm $SLURM_JOB_ID.hostfile
rm $SLURM_JOB_ID.hostfile.converted
