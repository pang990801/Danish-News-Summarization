import numpy as np
import torch
import datasets
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    AutoConfig
)
from torch.utils.data import DataLoader
import nltk

# Ensure the punkt tokenizer models are downloaded
nltk.download('punkt')

# Configuration and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/mt5-small"
local_model_path = "small_model"

# Function to load and prepare datasets
test_dataset = load_dataset("ScandEval/nordjylland-news-summarization-mini", split="test")
split_dataset = DatasetDict({'test': test_dataset})


# Function for tokenization
def tokenize_data(data, tokenizer):
    input_feature = tokenizer(data["input_text"], truncation=True, max_length=1024)
    label = tokenizer(data["target_text"], truncation=True, max_length=180)
    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": label["input_ids"],
    }


# Initialize tokenizer and model
mt5_tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path).to(device)

tokenized_dataset = split_dataset.map(
    lambda data: tokenize_data(data, mt5_tokenizer),
    remove_columns=['input_text', 'target_text', 'text_len', 'summary_len'],
    batched=True,
    batch_size=128
)

# DataLoader setup
data_collator = DataCollatorForSeq2Seq(
    mt5_tokenizer,
    model=model,
    return_tensors="pt"
)
sample_dataloader = DataLoader(
    tokenized_dataset["test"].with_format("torch"),
    collate_fn=data_collator,
    batch_size=2
)


# Generate and display predictions
def generate_and_display_predictions(dataloader, model, tokenizer, num_samples=5):
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        with torch.no_grad():
            preds = model.generate(
                batch["input_ids"].to(device),
                num_beams=2,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
                max_length=128
            )
        labels = batch["labels"]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print("***** Input's Text *****")
        print(split_dataset["test"]["input_text"][i])
        print("***** Summary Text (True Value) *****")
        print(text_labels[0])
        print("***** Summary Text (Generated Text) *****")
        print(text_preds[0])
        print('\n' + '-' * 50 + '\n')


# Run the generation and display function
generate_and_display_predictions(sample_dataloader, model, mt5_tokenizer)
