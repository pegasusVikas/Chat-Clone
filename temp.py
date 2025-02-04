'''from unsloth import FastLanguageModel
from transformers import TextStreamer
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    #device_map="auto",  # Auto-offload to CPU if GPU runs out
    offload_folder="offload_dir"  # Temporary disk storage
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

model.save_pretrained_merged("merged_model", tokenizer,max_shard_size="2GB" ,save_method = "merged_16bit",)
model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")'''


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-v0.3",load_in_4bit =True)
model = AutoModelForCausalLM.from_pretrained("unsloth/mistral-7b-v0.3",load_in_4bit =True)