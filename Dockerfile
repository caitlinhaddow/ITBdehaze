## CH Dissertation: New File added
## Primary method used for creating the code environment. Dockerfile is run by run_train.py and run_test.py

# Use the official Conda base image with Python 3.7
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set working directory
WORKDIR /ITBdehaze

# Install packages required for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install core requirements
RUN conda install -c conda-forge timm==1.0.15 -y
RUN pip install opencv-python scikit-image tensorboardx==2.6.2.2 yacs "numpy<2"

# Copy files into container
COPY . .

# Entry point for run code in run_train.py and run_test.py
ENTRYPOINT ["python"]