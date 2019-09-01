#!/bin/bash
#SBATCH --partition physical
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=2-00:00:00

module load scikit-learn

cd /home/zlp/SML

python3 train_spartan.py