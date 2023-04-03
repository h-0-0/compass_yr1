#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --job-name=CIFAR100
#SBATCH --nodes=1
#SBATCH --time=6:00:00
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
python main.py --data_name="CIFAR100" --model_name="CNN_VGG" --batch_size=64 --learning_rate=0.007 --epochs=500 --no-load_model --save_model
# python main.py --data_name="CIFAR100" --model_name="RN50_clip_FC_FF_NN" --batch_size=64 --learning_rate=0.005 --epochs=500 --no-load_model --save_model

# We use 10 tasks as done in the literature
# python main.py --data_name="CIFAR100" --model_name="CNN_VGG" --batch_size=64 --learning_rate=0.007 --epochs=5 --no-load_model --save_model --n_tasks=10 --init_n_tasks=1
# python main.py --data_name="CIFAR100" --model_name="RN50_clip_FC_FF_NN" --batch_size=64 --learning_rate=0.005 --epochs=5 --no-load_model --save_model --n_tasks=10 --init_n_tasks=1

echo End Time: $(date)
