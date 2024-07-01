#!/bin/bash
echo 'Running the script'
eval "$(conda shell.bash hook)"
# Activate your Conda environment
conda activate detr_clone

cd /media/homes/farjad/repos/detr_custom_dcmha/

current_directory=$(pwd)

echo "current directory is: $current_directory"

echo "executing script"


#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path /media/homes/alyoussef/datasets/mscoco2017/coco --epochs 2 --nheads 8 \
#> log_directory/logfile_epochs2_heads8.log 2>&1 &

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /media/homes/alyoussef/datasets/mscoco2017/coco --epochs 5 --nheads 4 \
#  --enc_layers 8 --dec_layers 8 --output_dir /home/alyoussef/farjad/detr/output/4_8e8d > log_directory/logfile_epochs5_heads4_8e8d.log 2>&1 &

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path /media/homes/farjad/data/coco --epochs 10 \
 --output_dir /media/homes/farjad/repos/detr_custom_dcmha/output --batch_size 2 > log_directory/gated_mha_4gpus.txt 2>&1 &


# OMP_NUM_THREADS=1 python tools/train_net.py --config-file configs/SOLOv2/R50_3x_1.yaml --num-gpus 8 OUTPUT_DIR training_dir/R50_3x_1≈ > logfile.log 2>&1 &
# python tools/train_net.py --config-file configs/SOLOv2/R50_3x_1.yaml --num-gpus 8 OUTPUT_DIR training_dir/R50_3x_1≈ >> logfile.log 2>&1 &
# python tools/train_net.py >> logfile.log 2>&1 &
# python ftest.py >> logfile.log 2>&1 &

echo "Script finished successfully!"
# Disown the process, which will keep it running even after you close the terminal
disown
