 ## Installation Guide

### If you have an NVIDIA GPU and want to use it for training:

#### 1. Install CUDA 12.1
- Go to [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- Download and install the CUDA Toolkit appropriate for your system

#### 2. Additional Requirements for Windows
- Install WSL
- Setup the requirments in the WSL

#### 3. Install Dependencies
##### Linux Setup
```sh
sudo apt-get install build-essential
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install cmake build-essential
```
##### Conda Environment
Use Conda to create virtual environment and install dependencies
##### Option 1:
```sh
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```
##### Or Option 2:
```
conda env create -f environment.yml
conda activate unsloth_env
```


## How to run
```sh
conda install ipdb pytest
