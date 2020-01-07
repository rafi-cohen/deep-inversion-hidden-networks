#!/bin/bash

###
# CS236781: Deep Learning
# nb-sbatch.sh
#
# This script runs a notebook as a slurm batch job.
# All arguments passed to this script are passed directly to the ipython.
#

###
# Example usage:
#
# Running a notebook
# ./nb-sbatch.sh main.py run-nb Part0_Intro.ipynb
#
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
QUEUE=rishonPartition
JOB_NAME="fakeShake"

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	-o 'slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
ipython $@

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

