# Use the official Conda base image with Python 3.7
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set working directory
WORKDIR /ITBdehaze

# required for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN conda install -c conda-forge timm==1.0.15 -y
RUN pip install opencv-python scikit-image tensorboardx==2.6.2.2 yacs "numpy<2"

COPY . .

# ENTRYPOINT ["bash", "-c"]
ENTRYPOINT ["python"]

# CMD ["python", "test.py", "--imagenet_model", "SwinTransformerV2", "--cfg", "configs/swinv2/swinv2_base_patch4_window8_256.yaml", "--model_save_dir", "./output_result", "--hazy_data", "NHNH2", "--cropping", "1"]

# TO BUILD IMAGE
# hare build -t ceh94/itb .

# # docker system prune ## run every so often to clear out system

# # TO RUN IMAGE FOR TESTING
# hare run --rm --gpus '"device=1,2"' --shm-size=128g \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/weights,target=/ITBdehaze/weights \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/output_result,target=/ITBdehaze/output_result \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/input_data,target=/ITBdehaze/input_data \
# ceh94/itb \
# test.py --imagenet_model SwinTransformerV2 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml --model_save_dir ./output_result --cropping 1 --datasets "NH" "NH2" "SMOKE_1600x1200_test" --weights "2025-04-24_13-29-41_DNHDenseRBm10_epoch01000.pkl" "2025-04-24_13-29-41_DNHDenseRBm10_epoch02000.pkl" "2025-04-24_13-29-41_DNHDenseRBm10_epoch03000.pkl" "2025-04-24_13-29-41_DNHDenseRBm10_epoch04000.pkl" "2025-04-24_13-29-41_DNHDenseRBm10_epoch05000.pkl" "2025-04-24_13-29-41_DNHDenseRBm10_epoch06000.pkl" "2025-04-24_13-29-41_DNHDenseRBm10_epoch07000.pkl" "2025-04-24_13-29-41_DNHDenseRBm10_epoch08000.pkl"

# hare run --rm --gpus '"device=6,7"' --shm-size=128g \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/weights,target=/ITBdehaze/weights \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/output_result,target=/ITBdehaze/output_result \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/input_data,target=/ITBdehaze/input_data \
# ceh94/itb \
# test.py --imagenet_model SwinTransformerV2 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml --model_save_dir ./output_result --cropping 1 --datasets "NH" "NH2" "SMOKE_1600x1200_test" --weights "2025-05-03_19-41-07_DNHDense_epoch01000.pkl" "2025-05-03_19-41-07_DNHDense_epoch02000.pkl" "2025-05-03_19-41-07_DNHDense_epoch03000.pkl" "2025-05-03_19-41-07_DNHDense_epoch04000.pkl" "2025-05-03_19-41-07_DNHDense_epoch05000.pkl" "2025-05-03_19-41-07_DNHDense_epoch06000.pkl" "2025-05-03_19-41-07_DNHDense_epoch07000.pkl" "2025-05-03_19-41-07_DNHDense_epoch08000.pkl" "2025-05-04_17-45-22_DNHDenseRB10_epoch01000.pkl" "2025-05-04_17-45-22_DNHDenseRB10_epoch02000.pkl" "2025-05-04_17-45-22_DNHDenseRB10_epoch03000.pkl" "2025-05-04_17-45-22_DNHDenseRB10_epoch04000.pkl" "2025-05-04_17-45-22_DNHDenseRB10_epoch05000.pkl" "2025-05-04_17-45-22_DNHDenseRB10_epoch06000.pkl" "2025-05-04_17-45-22_DNHDenseRB10_epoch07000.pkl" "2025-05-04_17-45-22_DNHDenseRB10_epoch08000.pkl"

# # TO RUN IMAGE FOR TRAINING
# hare run --rm --gpus '"device=2,1"' --shm-size=128g \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/weights,target=/ITBdehaze/weights \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/output_result,target=/ITBdehaze/output_result \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/train_input,target=/ITBdehaze/train_input \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/check_points,target=/ITBdehaze/check_points \
# ceh94/itb \
# train.py --imagenet_model SwinTransformerV2 --cropping 1 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml -train_batch_size 8 --model_save_dir check_points -train_epoch 8005 --datasets "DNHDense" "DNHDenseRB10"

