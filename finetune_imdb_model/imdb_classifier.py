# Imports from peft
from peft import LoraConfig, get_peft_model, AutoPeftModelForSequenceClassification

# Imports from transformers
from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    AutoTokenizer, 
    DataCollatorWithPadding
)

# Imports from datasets
from datasets import load_dataset, concatenate_datasets

# Imports from sklearn
from sklearn.metrics import accuracy_score

# Imports from other libraries
import numpy as np
import pandas as pd
import torch
import evaluate

# Tokenize and encode the dataset
def tokenize(batch, tokenizer, max_length=128):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128, return_tensors="pt")

def GetTokenizer(model_name):
   tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
   load_special_tokens(tokenizer)
   return tokenizer

def GetImdbDataSet(tokenizer):
   # Load the IMDB dataset
   dataset = load_dataset('imdb')
   dataset_shuffled = dataset.shuffle(seed=42) 

   tokenized_datasets = dataset_shuffled.map(lambda batch: tokenize(batch, tokenizer), batched=True)
   tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
   tokenized_datasets = tokenized_datasets.remove_columns(["text"])
   tokenized_datasets.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])

   tokenized_train_dataset = tokenized_datasets["train"].select(range(2000))
   tokenized_eval_dataset = tokenized_datasets["test"].select(range(1000))
   return tokenized_train_dataset, tokenized_eval_dataset

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def print_label_distributions(dataset, label_column="label", split=None):
    if split:  # Handle optional split
        dataset = dataset[split]

    # Get the label values as a list from the dataset
    label_values = dataset[label_column]

    # Create a Series from the list for value_counts() to work
    label_series = pd.Series(label_values)

    label_counts = label_series.value_counts()

    print("Label Distributions:")
    df = pd.DataFrame({'Label': label_counts.index, 'Count': label_counts.values})
    print(df)

    print("\nLabel Distribution Percentages:")
    percentage_df = pd.DataFrame({'Label': label_counts.index, 'Percentage': (label_counts / len(dataset) * 100).round(2)})
    print(percentage_df)

def print_tokenized_dataset_stats(tokenized_datasets):
    for split_name in tokenized_datasets.keys():
        print(f"\n\n*** Statistics for split: {split_name} ***")

        # Dataset Size
        num_examples = len(tokenized_datasets[split_name])
        print(f"Number of examples: {num_examples}")

        # Token Length Distribution (example)
        all_lengths = []
        for example in tokenized_datasets[split_name]:
            all_lengths.append(len(example["input_ids"]))

        avg_length = np.mean(all_lengths)
        max_length = np.max(all_lengths)
        min_length = np.min(all_lengths)

        print(f"Token lengths: Avg: {avg_length:.2f}, Max: {max_length}, Min: {min_length}")

        # Distribution Summary using Pandas
        lengths_series = pd.Series(all_lengths)
        length_distribution = lengths_series.value_counts()
        print("Token Length Distribution:")
        print(length_distribution)

def load_special_tokens(tokenizer):
    special_tokens_dict = {
        'bos_token': '<|startoftext|>',
        'eos_token': '<|endoftext|>',
        'unk_token': '<|unk|>',
        'pad_token': '<|pad|>'
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # Print vocabulary size
    print("Vocabulary size:", tokenizer.vocab_size)

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GetTokenizer(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model with num_labels=2 for binary classification and define the pad token, , pad_token_id=tokenizer.eos_token_id
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.resize_token_embeddings(len(tokenizer)) 

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load the IMDB dataset
imdb_train_dataset, imdb_test_dataset = GetImdbDataSet(tokenizer)

num_labels = imdb_test_dataset.features['labels'].num_classes
class_names = imdb_test_dataset.features["labels"].names
print(f"Label Count: {num_labels}")
print(f"Labels: {class_names}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./output/found_eval",
    
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    
    learning_rate=2e-4,
    num_train_epochs=1,
    weight_decay=0.01,
    
    logging_strategy="epoch",
    save_strategy="epoch",  # Save checkpoints every epoch
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=imdb_test_dataset,
    data_collator=data_collator,  # Add data collator
    compute_metrics=compute_metrics,
)

# Start evaluation
evaluation_results = trainer.evaluate()
print(evaluation_results)

print("####DONE WITH PART#1####\n")

### Part 2
# Load GPT-2 as the Base Language Model
print("Load GPT-2 as the Base Language Model")
base_model_name = "gpt2"
peft_model_name = 'gpt2-finetuned'
tokenizer_modified = 'gpt2-base-tokenizer-modified'

model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
print("Apply Low-Rank Adaptation (LoRA) for Efficient Fine-Tuning")
peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

print("Set Up Training Parameters")
training_args = TrainingArguments(
    output_dir="peft_imdb_model",              # Save fine-tuned model here
    per_device_train_batch_size=1,            # Number of examples per GPU in training
    per_device_eval_batch_size=1,             # Number of examples per GPU in evaluation
    learning_rate=2e-4,                        # Initial learning rate
    num_train_epochs=1,                        # Number of training epochs
    weight_decay=0.01,                         # Regularization to prevent overfitting
    logging_steps=5, 
    logging_strategy="steps",                 # Log training progress every epoch
    save_strategy="epoch",                     # Save model checkpoints every epoch
)

trainer = Trainer(
    model=peft_model,                           # The PEFT-modified model
    args=training_args,                         # Training configuration
    train_dataset=imdb_train_dataset,           # Training data
    eval_dataset=imdb_test_dataset,   # Evaluation data

    data_collator=data_collator,              # Handles padding in data batches
    compute_metrics=compute_metrics,          # Evaluation metric function
)

train_results = trainer.train()
print("train results")
print(evaluation_results)

evaluation_results = trainer.evaluate()
print("evaluation results")
print(evaluation_results)

tokenizer.save_pretrained(tokenizer_modified)
peft_model.save_pretrained(peft_model_name)

#Part3: Loading fine tuned model.
inference_model = AutoPeftModelForSequenceClassification.from_pretrained(peft_model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_modified)
trainer = Trainer(
    model=inference_model,                           # The PEFT-modified model
    args=training_args,                         # Training configuration
    train_dataset=imdb_train_dataset,           # Training data
    eval_dataset=imdb_test_dataset,   # Evaluation data

    data_collator=data_collator,              # Handles padding in data batches
    compute_metrics=compute_metrics,          # Evaluation metric function
)
evaluation_results = trainer.evaluate()
print("evaluation results based on loaded fine tuned model")
print(evaluation_results)