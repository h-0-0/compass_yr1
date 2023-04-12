#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=VGG_encoder
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --mem=5G
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
source ../../mypyenv/bin/activate
python encode.py --data_name="CIFAR10" --encoder_name="fVGG"

echo End Time: $(date)
