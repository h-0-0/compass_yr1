#!/bin/bash
# 
# 
#SBATCH --partition=gpu
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --mem=4G
#SBATCH --time=0-01:00:00
#SBATCH --signal=SIGHUP@90
#SBATCH --account MATH021322

# Define executable
export EXE=/bin/hostname

# Change into working directory
cd "${SLURM_SUBMIT_DIR}"

# Execute code
${EXE}

# Do some stuff
echo JOB ID: ${SLURM_JOBID}
echo Working Directory: $(pwd)

echo Start Time: $(date)

# Use this to create virtual env (keeping this here for my reference later on):
# python3 -m venv ./mypyenvb
# We activate the virtual environment
source mypyenv/bin/activate
python main.py --l_dim=256
# Can use below if you want to submit a job for execution in real time
# srun python3 train.py


echo End Time: $(date)