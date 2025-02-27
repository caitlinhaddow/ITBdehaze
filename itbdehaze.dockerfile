# Use the official Conda base image with Python 3.7
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set working directory
WORKDIR /ITBDehaze
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN conda install -c conda-forge timm==1.0.15 -y
RUN pip install opencv-python scikit-image tensorboardx==2.6.2.2 yacs "numpy<2"
#RUN conda init bash

# Copy environment file and install dependencies
#COPY itb_environment.yml .
#RUN conda env create -f itb_environment.yml && \
#    conda clean --all -y
#
# Activate the environment and set it as default
#RUN echo "conda activate venv_ITBDehaze" >> ~/.bashrc
#ENV PATH /opt/conda/envs/venv_ITBDehaze/bin:$PATH
#-v ~/ITBDehaze:/ITBDehaze
# Copy the model files and weights
COPY . .