#!/bin/bash

# BEFORE RUNNING: give execution permissions with chmod +x run_train.sh
# Then run with ./run_train.sh

set -e  # Exit on error
# set -x  # Echo commands for debugging

# BUILD image
hare build -t ceh94/itb .

# START THE CONTAINER and RUN TRAINING CODE
hare run --rm --gpus '"device=1,2"' --shm-size=128g \
  --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/weights,target=/ITBdehaze/weights \
  --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/output_result,target=/ITBdehaze/output_result \
  --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/input_training_data,target=/ITBdehaze/input_training_data \
  --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/check_points,target=/ITBdehaze/check_points \
  ceh94/itb \
  train.py \
  --imagenet_model SwinTransformerV2 \
  --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml \
  -train_batch_size 8 \
  --model_save_dir check_points \
  -train_epoch 0005 \
  --cropping 1 \
  --datasets "NHNH2" \
  --generate  # Run validation during training



## DATASETS: "NH" "NH2" "SMOKE_1600x1200_test"

# # RUN TRAINING CODE
# hare exec "$CONTAINER_ID" python train.py \
#   --imagenet_model SwinTransformerV2 \
#   --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml \
#   -train_batch_size 8 \
#   --model_save_dir check_points \
#   -train_epoch 0005 \
#   --cropping 1 \
#   --generate false
#   --datasets "NH2" \



