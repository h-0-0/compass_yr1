#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --job-name=gpujob
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
source ../mypyenv/bin/activate
# python main.py --data_name="CIFAR10" --model_name="FC_FF_NN" --batch_size=64 --learning_rate=0.01 --epochs=100 --no-load_model --save_model
# python main.py --data_name="CIFAR10" --model_name="CNN" --batch_size=64 --learning_rate=0.008 --epochs=100 --no-load_model --save_model
python main.py --data_name="CIFAR10" --model_name="RN50_clip_FC_FF_NN" --batch_size=64 --learning_rate=0.01 --epochs=100 --no-load_model --save_model


echo End Time: $(date)
