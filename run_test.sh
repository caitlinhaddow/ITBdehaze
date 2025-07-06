#!/bin/bash

# BEFORE RUNNING: give execution permissions with chmod +x run_train.sh
# Then run with ./run_train.sh

set -e  # Exit on error
# set -x  # Echo commands for debugging

# BUILD image
hare build -t ceh94/itb .

# START THE CONTAINER and RUN TEST CODE
hare run -rm --gpus '"device=1,2"' --shm-size=128g \
  --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/weights,target=/ITBdehaze/weights \
  --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/output_result,target=/ITBdehaze/output_result \
  --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/input_data,target=/ITBdehaze/input_data \
  ceh94/itb \
  test.py \
  --imagenet_model SwinTransformerV2 \
  --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml \
  --model_save_dir ./output_result \
  --cropping 1 \
  --datasets "NH2"

  ## DATASETS: "NH" "NH2" "SMOKE_1600x1200_test"



