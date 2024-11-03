#!/bin/bash

#SBATCH --job-name=train    # Specify your job name
#SBATCH --nodes=1  # Specify the number of nodes you want to use
#SBATCH --nodelist=oat05  # Specify the list of nodes you want to use
#SBATCH --time=2-00:00:00          # Allocate 1 days to the job
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --gres=gpu:1
#SBATCH --mem=128G 

python main.py