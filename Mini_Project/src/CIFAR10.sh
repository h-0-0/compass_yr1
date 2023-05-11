#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --job-name=CL-nfR50
#SBATCH --nodes=1
#SBATCH --time=16:00:00
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
nvidia-smi --query-gpu=name --format=csv,noheader

# Use this to create virtual env (keeping this here for my reference later on):
# python3 -m venv ./mypyenvb
# We activate the virtual environment
source ../mypyenv/bin/activate

# Commands for experiments we ran in batch scenario
# python main.py --data_name="CIFAR10" --model_name="RN50_clip_FC_FF_NN" --batch_size=128 --learning_rate=0.002 --epochs=100 --no-load_model --save_model
# No pretraining
# python main.py --data_name="CIFAR10" --model_name="CNN" --batch_size=128 --learning_rate=0.0001 --epochs=100 --no-load_model --save_model
# Pretrained and frozen
# python main.py --data_name="CIFAR10" --model_name="fVGG_FC_FF_NN" --batch_size=128 --learning_rate=0.0008 --epochs=100 --no-load_model --save_model
# python main.py --data_name="CIFAR10" --model_name="fResnet18_FC_FF_NN" --batch_size=128 --learning_rate=0.002 --epochs=100 --no-load_model --save_model
# python main.py --data_name="CIFAR10" --model_name="fResnet50_FC_FF_NN" --batch_size=128 --learning_rate=0.002 --epochs=100 --no-load_model --save_model
# python main.py --data_name="CIFAR10" --model_name="fAE_FC_FF_NN" --batch_size=128 --learning_rate=0.0025 --epochs=200 --no-load_model --save_model
# Pretrained and unfrozen
# python main.py --data_name="CIFAR10" --model_name="nfVGG_FC_FF_NN" --batch_size=128 --learning_rate=0.0008 --epochs=100 --no-load_model --save_model
# python main.py --data_name="CIFAR10" --model_name="nfResnet18_FC_FF_NN" --batch_size=128 --learning_rate=0.002 --epochs=100 --no-load_model --save_model
# python main.py --data_name="CIFAR10" --model_name="nfResnet50_FC_FF_NN" --batch_size=128 --learning_rate=0.002 --epochs=100 --no-load_model --save_model
# python main.py --data_name="CIFAR10" --model_name="nfAE_FC_FF_NN" --batch_size=128 --learning_rate=0.001 --epochs=200 --no-load_model --save_model

# Commands for experiments we ran in CL scenario, we use 5 tasks
# No pretraining
# python main.py --data_name="CIFAR10" --model_name="CNN" --batch_size=128 --learning_rate=0.0001 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 
# Pretrained and frozen
# python main.py --data_name="CIFAR10" --model_name="fVGG_FC_FF_NN" --batch_size=128 --learning_rate=0.00005 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 
# python main.py --data_name="CIFAR10" --model_name="fResnet18_FC_FF_NN" --batch_size=128 --learning_rate=0.00005 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 
# python main.py --data_name="CIFAR10" --model_name="fResnet50_FC_FF_NN" --batch_size=128 --learning_rate=0.00005 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 
# python main.py --data_name="CIFAR10" --model_name="fAE_FC_FF_NN" --batch_size=128 --learning_rate=0.00005 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2
# Pretrained and unfrozen
# python main.py --data_name="CIFAR10" --model_name="nfVGG_FC_FF_NN" --batch_size=128 --learning_rate=0.00001 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2
# python main.py --data_name="CIFAR10" --model_name="nfResnet18_FC_FF_NN" --batch_size=128 --learning_rate=0.00001 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 
python main.py --data_name="CIFAR10" --model_name="nfResnet50_FC_FF_NN" --batch_size=128 --learning_rate=0.00001 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 
# python main.py --data_name="CIFAR10" --model_name="nfAE_FC_FF_NN" --batch_size=128 --learning_rate=0.00005 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 

# Commands for experiments we ran in CL scenario, with reset of fully connected layers, we use 5 tasks
# Pretrained and frozen
# python main.py --data_name="CIFAR10" --model_name="fVGG_FC_FF_NN_reset" --batch_size=128 --learning_rate=0.00005 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 --reset_fc
# python main.py --data_name="CIFAR10" --model_name="fResnet18_FC_FF_NN_reset" --batch_size=128 --learning_rate=0.00005 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 --reset_fc
# python main.py --data_name="CIFAR10" --model_name="fAE_FC_FF_NN_reset" --batch_size=128 --learning_rate=0.00005 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 --reset_fc
# Pretrained and unfrozen
# python main.py --data_name="CIFAR10" --model_name="nfVGG_FC_FF_NN_reset" --batch_size=128 --learning_rate=0.00001 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 --reset_fc
# python main.py --data_name="CIFAR10" --model_name="nfResnet18_FC_FF_NN_reset" --batch_size=128 --learning_rate=0.00001 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 --reset_fc
# python main.py --data_name="CIFAR10" --model_name="nfAE_FC_FF_NN_reset" --batch_size=128 --learning_rate=0.00005 --epochs=15 --no-load_model --save_model --n_tasks=5 --init_inc=2 --reset_fc

echo End Time: $(date)
