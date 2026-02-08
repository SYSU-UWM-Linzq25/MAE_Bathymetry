#!/bin/bash

#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --output=/home/uwm/maopuxu/MAE_Topography_Reconstruction/MAE-Topography/output/mae_finetune-%j.out

set -e

cd /home/uwm/maopuxu

source /home/uwm/maopuxu/miniconda3/bin/activate

conda activate mae_test


python /home/uwm/maopuxu/MAE_Topography_Reconstruction/MAE-Topography/mae/main_finetune.py --model vit_large_patch16 --finetune /tank/data/SFS/xinyis/shared/data/mae/output_dir/checkpoint-399.pth --epochs 100 --data_path /home/uwm/maopuxu/MAE_Topography_Reconstruction/MAE-Topography/samples --output_dir /tank/data/SFS/xinyis/shared/data/mae/finetune_output_dir --log_dir /tank/data/SFS/xinyis/shared/data/mae
