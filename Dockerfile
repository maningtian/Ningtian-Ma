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

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy only the requirements file first, to leverage Docker caching
COPY requirements.txt ./

# Install project dependencies
RUN python3 -m pip install -r requirements.txt

# Create the cache directory
RUN mkdir -p hf_cache

# Set ownership and permissions for writeable directories
RUN useradd app
RUN chown app:app hf_cache
RUN chmod 700 hf_cache

# Set environment variables
ENV HF_HOME=/usr/local/app/hf_cache

# Copy the rest of the project files
COPY data ./data
COPY models ./models
COPY code ./code

EXPOSE 5000

USER app

CMD ["nvidia-smi"]
