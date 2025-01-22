from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Path to your fine-tuned model directory
model_path = "friend_chat_model"  # Directory containing model.safetensors

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Load the model
model = GPT2LMHeadModel.from_pretrained(model_path)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

while(True):
    # Example input
    input_text = input("You:")
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate text
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1,
            temperature=0.7
        )

    # Decode and print the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
