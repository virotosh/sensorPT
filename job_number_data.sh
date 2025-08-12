#!/bin/bash
#SBATCH --job-name=vuong
#SBATCH --account=project_2014260
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1

source /projappl/project_2014260/mypythonproject/bin/activate
srun python experiment_number_data.py
