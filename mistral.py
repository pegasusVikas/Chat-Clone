'''
import sys
import subprocess
import torch
def check_gpu_status():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        nvidia_output = subprocess.check_output(['nvidia-smi'], shell=True)
        print("NVIDIA-SMI output:")
        print(nvidia_output.decode())
    except:
        print("nvidia-smi not found - GPU driver may not be installed")

if __name__ == "__main__":
    check_gpu_status()

'''

from ast import Import


def main():
    from unsloth import FastLanguageModel
    import torch
    import json
    import os
    from datasets import Dataset
    import sys
    from unsloth.chat_templates import get_chat_template
    from transformers import TextStreamer
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    from unsloth.chat_templates import get_chat_template
    from transformers import TextStreamer
    HUGGING_FACE_TOKEN=os.getenv("HUGGING_FACE_TOKEN")
    #import os
    #os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
        "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

        "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
    ] # More models at https://huggingface.co/unsloth
    print("Choosing model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # or choose "unsloth/Llama-3.2-1B-Instruct" or "unsloth/Llama-3.2-3B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        token = HUGGING_FACE_TOKEN, # Fetch token from environment variable
    )
    print("Downloading model")
    """We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    print("Split model")
    """<a name="Data"></a>
    ### Data Prep
    We now use the `Llama-3.1` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. But we convert it to HuggingFace's normal multiturn format `("role", "content")` instead of `("from", "value")`/ Llama-3 renders multi turn conversations like below:

    ```
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>

    Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Hey there! How are you?<|eot_id|><|start_header_id|>user<|end_header_id|>

    I'm great thanks!<|eot_id|>
    ```

    We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3` and more.
    """


    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    pass

    '''
    converting to Hugging Face Generic format
    '''
    def convert_format(dataset):
        converted_data = []
        
        for item in dataset["conversations"]:
            conversation = [
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]}
            ]
            converted_data.append({"conversations": conversation})
        return Dataset.from_list(converted_data)
    pass

    try:
        dataset = Dataset.from_json("training_data.json")
        print(f"Successfully loaded {len(dataset)} examples")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    """We now use `standardize_sharegpt` to convert ShareGPT style datasets into HuggingFace's generic format. This changes the dataset from looking like:
    ```
    {"from": "system", "value": "You are an assistant"}
    {"from": "human", "value": "What is 2+2?"}
    {"from": "gpt", "value": "It's 4."}
    ```
    to
    ```
    {"role": "system", "content": "You are an assistant"}
    {"role": "user", "content": "What is 2+2?"}
    {"role": "assistant", "content": "It's 4."}
    ```
    """
    dataset = convert_format(dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    """We look at how the conversations are structured for item 5:"""

    dataset[5]["conversations"]

    """And we see how the chat template transformed these conversations.

    **[Notice]** Llama 3.1 Instruct's default chat template default adds `"Cutting Knowledge Date: December 2023\nToday Date: 26 July 2024"`, so do not be alarmed!
    """

    dataset[5]["text"]

    """<a name="Train"></a>
    ### Train the model
    Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support TRL's `DPOTrainer`!
    """

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )

    """We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs."""


    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    """We verify masking is actually done:"""

    tokenizer.decode(trainer.train_dataset[5]["input_ids"])

    space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

    """We can see the System and Instruction prompts are successfully masked!"""

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    """<a name="Inference"></a>
    ### Inference
    Let's run the model! You can change the instruction and input - leave the output blank!

    **[NEW] Try 2x faster inference in a free Colab for Llama-3.1 8b Instruct [here](https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)**

    We use `min_p = 0.1` and `temperature = 1.5`. Read this [Tweet](https://x.com/menhguin/status/1826132708508213629) for more information on why.
    """



    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                            temperature = 1.5, min_p = 0.1)
    tokenizer.batch_decode(outputs)

    """ You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!"""

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")


    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)

    """<a name="Save"></a>
    ### Saving, loading finetuned models
    To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

    **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
    """
    print("Saving Lora Model Locally")
    model.save_pretrained("lora_model")  # Local saving
    tokenizer.save_pretrained("lora_model")
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

    """Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:"""
    print("Loading Lora Adapters")
    if True:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": "Describe a tall tower in the capital of France."},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")


    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)

    """You can also use Hugging Face's `AutoModelForPeftCausalLM`. Only use this if you do not have `unsloth` installed. It can be hopelessly slow, since `4bit` model downloading is not supported, and Unsloth's **inference is 2x faster**."""

    if False:
        # I highly do NOT suggest - use Unsloth if possible
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        model = AutoPeftModelForCausalLM.from_pretrained(
            "lora_model", # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit = load_in_4bit,
        )
        tokenizer = AutoTokenizer.from_pretrained("lora_model")

    """### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
    """

    # Merge to 16bit
    if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
    if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

    # Merge to 4bit
    if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
    if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

    # Just LoRA adapters
    if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
    if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

    """### GGUF / llama.cpp Conversion
    To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

    Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
    * `q8_0` - Fast conversion. High resource use, but generally acceptable.
    * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
    * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

    [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)
    """

    # Save to 8bit Q8_0
    if False: model.save_pretrained_gguf("model", tokenizer,)
    # Remember to go to https://huggingface.co/settings/tokens for a token!
    # And change hf to your username!
    if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")
    print("Saving to GGUF")
    # Save to 16bit GGUF
    if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
    if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

    # Save to q4_k_m GGUF
    if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "hf/model", # Change hf to your username!
            tokenizer,
            quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
            token = "", # Get a token at https://huggingface.co/settings/tokens
        )

    """Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)

    And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

    Some other links:
    1. Llama 3.2 Conversational notebook. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)
    2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
    3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
    6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!

    <div class="align-center">
    <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
    <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

    Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
    </div>

    """
if __name__ == '__main__':
    main()