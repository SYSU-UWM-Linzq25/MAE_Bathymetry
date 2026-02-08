#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH -p HyrdoIntel

#SBATCH --output=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/log/mae_initial-%j.out

set -e

cd /home/uwm/zequnlin

source /home/uwm/zequnlin/miniconda3/bin/activate

conda activate /tank/data/SFS/xinyis/shared/data/conda_envs/envs/mae_test_new

python3 /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/mae/main_pretrain.py --input_size 336 --data_path /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Data/Center_Test --output_dir /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/output/Center_Test --log_dir /tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/log
