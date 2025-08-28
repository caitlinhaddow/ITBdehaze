#!/bin/bash

## CH Dissertation: New File added
## Runs code needed to test the model. Assumes code is on Hex in faster0/ceh94/ITBdehaze - if not needs replacing.
## Datasets and weight files to batch test should be listed after the appropriate parameter as strings separated by spaces only.
## BEFORE RUNNING: give execution permissions with chmod +x run_test.sh
## Then run with ./run_test.sh

set -e  # Exit on error
# set -x  # Echo commands for debugging

# BUILD image
hare build -t ceh94/itb .

# START THE CONTAINER and RUN TEST CODE
hare run --rm --gpus '"device=1,2"' --shm-size=128g \
  --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/weights,target=/ITBdehaze/weights \
  --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/output_result,target=/ITBdehaze/output_result \
  --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/input_data,target=/ITBdehaze/input_data \
  ceh94/itb \
  test.py \
  --imagenet_model SwinTransformerV2 \
  --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml \
  --model_save_dir ./output_result \
  --cropping 1 \
  --datasets "NH" "NH2" "SMOKE_1600x1200_test" \
  --weights "2025-07-30_08-34-46_DNHDenseRBm5_epoch01000.pkl" "2025-07-30_08-34-46_DNHDenseRBm5_epoch02000.pkl"



