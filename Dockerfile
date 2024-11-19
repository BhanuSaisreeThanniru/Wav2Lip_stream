# Use NVIDIA CUDA base image with Python support for GPU-enabled ML workloads
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libgl1-mesa-glx \
    python3.8 \
    python3.8-distutils \
    wget \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py && rm get-pip.py

# Set Python aliases to use Python 3.8 explicitly
RUN ln -s /usr/bin/python3.8 /usr/bin/python && \
    ln -s /usr/local/bin/pip /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy application files into the image
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port (Streamlit uses port 8501)
EXPOSE 8501

# Command to run the application
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501"]
# "--server.address=0.0.0.0"