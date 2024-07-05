from common import tokenize, compute_metrics, print_label_distributions, print_tokenized_dataset_stats, compute_metrics, load_special_tokens, GetTokenizer, GetImdbDataSet
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np
import pandas as pd

print("Step#: Load GPT-2 as the Base Language Model")
base_model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(base_model_name, num_labels=2)
tokenizer = GetTokenizer(base_model_name)
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenizer.pad_token = tokenizer.eos_token

print("Step#: Apply Low-Rank Adaptation (LoRA) for Efficient Fine-Tuning")
peft_config = LoraConfig(
    r=8,                      # Rank of the low-rank matrices (affects model size/efficiency)
    lora_alpha=16,            # Scaling factor for LoRA attention
    lora_dropout=0.05,        # Dropout rate for LoRA layers
    bias="none",              # Bias type for LoRA parameters
    task_type="causal_lm"     # Specify the task type for LoRA
)
model = get_peft_model(model, peft_config)
model.resize_token_embeddings(len(tokenizer)) 

print("Step#: Load the IMDB dataset")
imdb_train_dataset, imdb_test_dataset = GetImdbDataSet(tokenizer)


print("####Printing train dataset")
for batch in imdb_train_dataset:
    print(batch.keys())  # Print keys to check for 'input_ids', 'labels', etc.
    inputs = batch['input_ids']
    targets = batch['labels']
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
    outputs = model(inputs)

print("####Printing test dataset")
for batch in imdb_test_dataset:
    print(batch.keys())  # Print keys to check for 'input_ids', 'labels', etc.
    inputs = batch['input_ids']
    targets = batch['labels']
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
    outputs = model(inputs)

# Manage Variable Sequence Lengths
#from transformers import DataCollatorForLanguageModeling 
#print("Step#: Manage Variable Sequence Lengths")
#data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

# Set Up Training Parameters
print("Step#: Set Up Training Parameters")
training_args = TrainingArguments(
    output_dir="peft_imdb_model",             # Save fine-tuned model here
    per_device_train_batch_size=1,            # Number of examples per GPU in training
    per_device_eval_batch_size=1,             # Number of examples per GPU in evaluation
    learning_rate=2e-4,                       # Initial learning rate
    num_train_epochs=1,                       # Number of training epochs
    weight_decay=0.01,                        # Regularization to prevent overfitting
    logging_steps=5, 
    logging_strategy="steps",                 # Log training progress every epoch
    save_strategy="epoch",                    # Save model checkpoints every epoch
)

trainer = Trainer(
    model=model,                             # The PEFT-modified model
    args=training_args,                      # Training configuration
    train_dataset=imdb_train_dataset,        # Training data
    eval_dataset=imdb_test_dataset,          # Evaluation data

    data_collator=data_collator,             # Handles padding in data batches
    compute_metrics=compute_metrics,         # Evaluation metric function
)

trainer.train()
