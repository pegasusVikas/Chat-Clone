import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json

def load_chat_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return Dataset.from_dict({
        'input_text': [item['input'] for item in data['conversations']],
        'output_text': [item['output'] for item in data['conversations']]
    })

def tokenize_conversations(examples, tokenizer):
    conversations = [
        f"User: {input_text}\nFriend: {output_text}" 
        for input_text, output_text in zip(examples['input_text'], examples['output_text'])
    ]
    
    result = tokenizer(
        conversations,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt"
    )
    
    # Set labels same as input_ids for casual language modeling
    result["labels"] = result["input_ids"].clone()
    
    return result

def train_model():
    # Initialize model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Load and prepare dataset
    dataset = load_chat_data('training_data.json')
    train_val = dataset.train_test_split(test_size=0.1)

    # Tokenize datasets
    tokenized_train = train_val['train'].map(
        lambda x: tokenize_conversations(x, tokenizer),
        batched=True,
        remove_columns=train_val['train'].column_names
    )
    tokenized_val = train_val['test'].map(
        lambda x: tokenize_conversations(x, tokenizer),
        batched=True,
        remove_columns=train_val['test'].column_names
    )

    # Configure training
    training_args = TrainingArguments(
        output_dir="./friend_chat_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=100,
        save_steps=200,
        warmup_steps=100,
        eval_strategy="steps",
        logging_dir='./logs',
        learning_rate=5e-5
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )
    )

    # Train and save
    trainer.train()
    model.save_pretrained("./friend_chat_model")
    tokenizer.save_pretrained("./friend_chat_model")

if __name__ == "__main__":
    train_model()