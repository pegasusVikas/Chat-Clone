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

    HUGGING_FACE_TOKEN=os.getenv("HUGGING_FACE_TOKEN") # Use Hugging Face token if you want to use Hugging Face models
    max_seq_length = 2048 
    dtype = None 
    load_in_4bit = True 

    
    fourbit_models = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    
        "unsloth/Mistral-Small-Instruct-2409",    
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",           
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",            

        "unsloth/Llama-3.2-1B-bnb-4bit",          
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" 
    ] # More models at https://huggingface.co/unsloth
    print("Choosing model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-3B-Instruct",
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
        lora_dropout = 0, 
        bias = "none",    
       
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None,
    )
   
    """
    ### Data Prep
    We now use the `Llama-3.1` format for conversation style finetunes

    ```
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>

    Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Hey there! How are you?<|eot_id|><|start_header_id|>user<|end_header_id|>

    I'm great thanks!<|eot_id|>
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
    
    # Loading our training data
    try:
        dataset = Dataset.from_json("training_data.json")
        print(f"Successfully loaded {len(dataset)} examples")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    dataset = convert_format(dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True,)


    dataset[5]["conversations"]

    dataset[5]["text"]

    """
    ### Train the model
    """

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, 
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
            report_to = "none", 
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

    """
    ### Saving finetuned models
    This ONLY saves the LoRA adapters, and not the full model.
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


    """### Saving to float16 """

    # Merge to 16bit
    if True:
        print("Saving Merged Model")
        model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
        print("Saved Merged Model")
if __name__ == '__main__':
    main()