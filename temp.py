from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize tokenizer with padding token
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Initialize model with pad token ID
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

while(True):
    prompt = input("You :")

    # Get input IDs and attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )

    # Generate with attention mask
    gen_tokens = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=True,
        temperature=0.9,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id
    )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)