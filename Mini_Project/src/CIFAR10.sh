#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=CIFAR10_fVGG_FC_FF_NN
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
# python main.py --data_name="CIFAR10" --model_name="SGDm_CNN_VGG16_00001" --batch_size=128 --learning_rate=0.00001 --epochs=100 --no-load_model --save_model --optimizer_type="SGD_momentum"
# python main.py --data_name="CIFAR10" --model_name="SGD_RN50_clip_FC_FF_NN" --batch_size=128 --learning_rate=0.025 --epochs=100 --no-load_model --save_model --optimizer_type="SGD"
python main.py --data_name="CIFAR10" --model_name="fVGG_FC_FF_NN" --batch_size=128 --learning_rate=0.002 --epochs=100 --no-load_model --save_model

# We use 5 tasks as mentioned in latent replay paper
# python main.py --data_name="CIFAR10" --model_name="reset_CNN_VGG16_e10" --batch_size=128 --learning_rate=0.0001 --epochs=10 --no-load_model --save_model --n_tasks=5 --init_inc=2 
# python main.py --data_name="CIFAR10" --model_name="RN50_clip_FC_FF_NN_e10" --batch_size=128 --learning_rate=0.002 --epochs=10 --no-load_model --save_model --n_tasks=5 --init_inc=2

# python main.py --data_name="CIFAR10" --model_name="SGDm_CNN_VGG16_e1" --batch_size=128 --learning_rate=0.0001 --epochs=1 --no-load_model --save_model --n_tasks=5 --init_inc=2 --optimizer_type="SGD_momentum"
# python main.py --data_name="CIFAR10" --model_name="SGD_RN50_clip_FC_FF_NN_e10" --batch_size=128 --learning_rate=0.025 --epochs=10 --no-load_model --save_model --n_tasks=5 --init_inc=2 --optimizer_type="SGD"

echo End Time: $(date)
