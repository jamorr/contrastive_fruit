FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0
# # Install any python packages you need
COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install lightly torch torchvision torchaudio matplotlib scikit-learn

# Set the working directory
WORKDIR /app

# Create a volume at /app/data
VOLUME /app/data

# Copy files from the host to the container
COPY . /app

# Set the entrypoint
ENTRYPOINT [ "bash" ]