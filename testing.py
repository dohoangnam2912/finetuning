from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the saved model and tokenizer
save_directory = "./saved_model"
model = AutoModelForCausalLM.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

# Test the model
input_text = "Instruction: Hãy cho tôi ba mẹo để giữ sức khỏe.\nResponse:"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
