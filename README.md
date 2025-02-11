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
```sh
conda env create -f environment.yml
conda activate unsloth_env
```


## How to run
#### 1. Activate env
```sh
conda activate unsloth_env
```
#### 2. Export WhatsApp chats and put it in the current directory with file name as **whatsapp_chat.txt**
#### 3. Run WhatsAppConverter.py
It converts the **whatsapp_chat.txt** into training data ( training_data.json )
```sh
python WhatsAppConverter.py
```
your data is ready now you can use one of the 3 trainers

## Available Trainers

### Mistral 7B

- Uses **Unsloth** with **LoRA fine-tuning**
- **Fast**
- Uncensored 

#### Run:

```sh
python mistral.py
```

---

### Llama-3.2-3B

- Uses **Unsloth** with **LoRA fine-tuning**
- **Very fast** (since it's a 3B model)
- Censored 

#### Run:

```sh
python llama.py
```

---

### GPT-2

- **Supervised fine-tuning** using **Transformers**
- **Slow**
- **Not recommended** (poor results achieved by me)

#### Run:

```sh
python Trainer_GPT2.py
```

### Output
---
It creates LoRA adapter and merged model. Merged model is your fine tunned model.
- LoRA adapter (light weight adapters)
- Merged Model (Lora Adapter + base model = fine tuned model)

You can use this model and run it through **transformers**. If you want to access it in a more simple way you can use **Ollama**.

### Run it in Ollama

Ollama supports GGUF model. Convert it to GGUF model to access the fine tuned model through ollama.
#### 1. Convert to GGUF
Go to the working directory and clone this repo.
``` sh
git clone --recursive https://github.com/ggerganov/llama.cpp
make clean -C llama.cpp
make all -j -C llama.cpp
pip install gguf protobuf
```
After cloning the project run this to convert your merged model to GGUF model

``` sh
python llama.cpp/convert_hf_to_gguf.py merged_model --outfile myModel.gguf --outtype f16
```

#### 2. Create Modelfile

Create Modelfile and write the following commands in it
``` txt
FROM ./myclone.gguf
SYSTEM You are an assistant who responds to user.
```
Let's create our custom model in Ollama and Run it

``` cmd
Ollama create <model-name> -f Modelfile
Ollama run <model-name>
```


