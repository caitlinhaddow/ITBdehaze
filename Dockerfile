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

CMD ["python", "test.py", "--imagenet_model", "SwinTransformerV2", "--cfg", "configs/swinv2/swinv2_base_patch4_window8_256.yaml", "--model_save_dir", "./output_result", "--hazy_data", "NHNH2", "--cropping", "1"]



# TO BUILD IMAGE
# hare build -t ceh94/itb .

# # docker system prune ## run every so often to clear out system

# # TO RUN IMAGE FOR TESTING
# hare run --rm --gpus '"device=5,6"' --shm-size=128g \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/weights,target=/ITBdehaze/weights \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/output_result,target=/ITBdehaze/output_result \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/input_data,target=/ITBdehaze/input_data \
# ceh94/itb \
# test.py --imagenet_model SwinTransformerV2 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml --model_save_dir ./output_result --hazy_data NHNH2 --cropping 1

# # TO RUN IMAGE FOR TRAINING
# hare run --rm --gpus '"device=5,6"' --shm-size=128g \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/weights,target=/ITBdehaze/weights \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/output_result,target=/ITBdehaze/output_result \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/data,target=/ITBdehaze/data \
# --mount type=bind,source=/mnt/faster0/ceh94/ITBdehaze/check_points,target=/ITBdehaze/check_points \
# ceh94/itb \
# train.py --data_dir data --imagenet_model SwinTransformerV2 --cropping 1 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml -train_batch_size 8 --model_save_dir check_points -train_epoch 8005