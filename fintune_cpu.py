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

device = torch.device('cpu')  # Change to CPU

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

dataset = Dataset.from_dict({'text': [entry['text'] for entry in data]})
split_datasets = dataset.train_test_split(test_size=0.1)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

tokenized_datasets = split_datasets.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    gradient_accumulation_steps=8,
)

trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

trainer.train()

save_directory = "./saved_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
