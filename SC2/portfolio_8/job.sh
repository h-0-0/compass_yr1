#!/bin/bash
# 
# 
#SBATCH --partition=gpu
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --mem=4G
#SBATCH --time=0-04:00:00
#SBATCH --account MATH021322
#SBATCH --job-name=AE-2048

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
python main.py --l_dim=2048 --ae
# Can use below if you want to submit a job for execution in real time
# srun python3 train.py


echo End Time: $(date)
