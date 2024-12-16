from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import os
import json
import torch

# Load pre-trained model and tokenizer
model_name = 'EleutherAI/gpt-neo-1.3B'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Path to the folder containing JSON files
data_folder = './data'
data = []
for filename in os.listdir(data_folder):
    if filename.endswith('.json'):
        with open(os.path.join(data_folder, filename)) as f:
            file_data = json.load(f)
            for entry in file_data:
                instruction = entry.get('instruction', '')
                output = entry.get('output', '')
                if instruction and output:
                    combined_text = f"Instruction: {instruction}\nResponse: {output}"
                    data.append({'text': combined_text})

# Convert data into a Dataset
dataset = Dataset.from_dict({'text': [entry['text'] for entry in data]})

# Split into train and validation datasets
split_datasets = dataset.train_test_split(test_size=0.1)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = split_datasets.map(tokenize_function, batched=True)

# Define training arguments with reduced batch size and mixed precision
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduced batch size
    per_device_eval_batch_size=1,   # Reduced batch size
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision training
)

# Create Trainer instance
trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# Set environment variable to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Train the model
trainer.train()

trainer.train()

save_directory = "./saved_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
