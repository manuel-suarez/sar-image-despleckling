#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=sar-image-despeckling
#SBATCH --ntasks=1
#SBATCH --output=result.log
#SBATCH --mem=0
#SBATCH --time=0
$HOME/anaconda3/envs/tf28/bin/python main.py > result.out
