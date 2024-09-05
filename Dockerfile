# Use Ubuntu 22.04 LTS as the base image
FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    git-lfs \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /usr/local/app

# Create the cache directory
RUN mkdir -p cache

# Set environment variables
ENV TRANSFORMERS_CACHE=/usr/local/app/cache

COPY requirements.txt ./

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install project dependencies
RUN python3 -m pip install -r requirements.txt

COPY code ./code
COPY data ./data
COPY models ./models

RUN useradd app
USER app

# CMD ["nvidia-smi"]
