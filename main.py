import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import json

def prepare_chat_data(chat_file):
    """
    Prepare WhatsApp chat data for training.
    Expected format: JSON file with message pairs
    """
    with open(chat_file, 'r',encoding='utf-8') as f:
        chat_data = json.load(f)
    
    # Convert to format expected by transformers
    dataset = Dataset.from_dict({
        'input_text': [item['input'] for item in chat_data['conversations']],
        'output_text': [item['output'] for item in chat_data['conversations']]
    })
    return dataset

def tokenize_data(examples, tokenizer):
    """Tokenize the input and output texts"""
    # Combine input and output with appropriate markers
    prompts = [f"<|input|>{inp}<|output|>{out}<|end|>" 
              for inp, out in zip(examples['input_text'], examples['output_text'])]
    
    # Tokenize with padding
    return tokenizer(
        prompts,
        truncation=True,
        padding='max_length',
        max_length=512
    )

def train_chat_model(model_name="gpt2", chat_file="training_data.json", output_dir="./chat_model"):
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Prepare dataset
    dataset = prepare_chat_data(chat_file)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_data(x, tokenizer),
        batched=True
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                  'attention_mask': torch.stack([f['attention_mask'] for f in data])}
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

# Function to generate responses using the trained model
def generate_response(model, tokenizer, input_text, max_length=100):
    # Prepare input
    prompt = f"<|input|>{input_text}<|output|>"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Generate response
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|output|>")[-1].split("<|end|>")[0].strip()
    
    return response

def ping():
    print("pong!");

ping()
train_chat_model()
ping()