from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np
import evaluate
import torch

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

   tokenized_train_dataset = tokenized_datasets["train"].select(range(1000))
   tokenized_eval_dataset = tokenized_datasets["test"].select(range(200))
   return tokenized_train_dataset, tokenized_eval_dataset

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def print_label_distributions(dataset, label_column="label", split=None):
    """
    Prints the distribution of labels in a Hugging Face dataset.

    Args:
        dataset (Dataset): The Hugging Face dataset to analyze.
        label_column (str, optional): The name of the column containing the labels. Defaults to "label".
        split (str, optional): The split of the dataset to use (e.g., "train", "test"). Defaults to None (use entire dataset).
    """

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
