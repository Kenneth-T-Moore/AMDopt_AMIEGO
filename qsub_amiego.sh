#!/bin/bash
#
# Use the ompi parallel environment, with X processors:
#$ -pe ompi 140
#
# The name of the job:
#$ -N test_AMIEGO
#
# Start from the current working directory:
#$ -cwd
#
# Set the shell:
#$ -S /bin/bash
#
# Retain the existing environment (except PATH, see below):
#$ -V
#
# Output and error filenames (optional):
#$ -o $JOB_NAME-$JOB_ID.output
#$ -e $JOB_NAME-$JOB_ID.error
#

module load engapps openmpi/1.8.3/gnu/64bit

export USE_PROC_FILES=1

source ~/Work/virtual_isolated_python/bin/activate
source ~/Work/mdolab_rolled_back/setup_mdolab.sh

cd ~/Work/mdolab_rolled_back/AMDopt_AMIEGO
mpirun -n 128 python amiego_run.py
