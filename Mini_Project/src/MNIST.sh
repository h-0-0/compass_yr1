#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=MNIST
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem=4G
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
/usr/bin/nvidia-smi
# Use this to create virtual env (keeping this here for my reference later on):
# python3 -m venv ./mypyenvb
# We activate the virtual environment
source ../mypyenv/bin/activate
# python main.py --data_name="MNIST" --model_name="FC_FF_NN" --batch_size=64 --learning_rate=0.007 --epochs=100 --load_model --save_model --optimizer_type="SGD"
# python main.py --data_name="MNIST" --model_name="CNN" --batch_size=64 --learning_rate=0.005 --epochs=50 --no-load_model --save_model --optimizer_type="SGD"

# Split MNIST, 5 tasks
python main.py --data_name="MNIST" --model_name="FC_FF_NN" --batch_size=64 --learning_rate=0.007 --epochs=10 --no-load_model --save_model --n_tasks=5 --init_inc=2 --optimizer_type="SGD"
python main.py --data_name="MNIST" --model_name="CNN" --batch_size=64 --learning_rate=0.005 --epochs=10 --no-load_model --save_model --n_tasks=5 --init_inc=2 --optimizer_type="SGD"

echo End Time: $(date)
