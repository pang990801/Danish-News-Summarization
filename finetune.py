import nltk
import numpy as np
import torch
import datasets
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoConfig,
)
from datasets import DatasetDict

import os
from transformers import AutoModelForSeq2SeqLM

model_name = "google/mt5-large"


train_dataset = load_dataset('csv', data_files='datasets/train_dataset.csv')['train']
validation_dataset = load_dataset('csv', data_files='datasets/validation_dataset.csv')['train']
test_dataset = load_dataset('csv', data_files='datasets/test_dataset.csv')['train']


split_dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

mt5_tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_data(data):
  input_feature = mt5_tokenizer(data["body"], truncation=True, max_length=1024)
  label = mt5_tokenizer(data["summary"], truncation=True, max_length=128)
  return {
    "input_ids": input_feature["input_ids"],
    "attention_mask": input_feature["attention_mask"],
    "labels": label["input_ids"],
  }

tokenized_dataset = split_dataset.map(
  tokenize_data,
  remove_columns=["id", "body", "summary"],
  batched=True,
  batch_size=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mt5_config = AutoConfig.from_pretrained(
  model_name,
  min_length=9,
  max_length=128,
  length_penalty=0.8,
  no_repeat_ngram_size=3,
  num_beams=4,
  dropout_rate=0.1
)
model = (AutoModelForSeq2SeqLM
         .from_pretrained(model_name, config=mt5_config)
         .to(device))

data_collator = DataCollatorForSeq2Seq(
  mt5_tokenizer,
  model=model,
  return_tensors="pt")


training_args = Seq2SeqTrainingArguments(
  output_dir = "mt5-summarize-large",
  log_level = "error",
  num_train_epochs = 20,
  learning_rate = 0.0003,
  lr_scheduler_type = "polynomial",# str: "linear" or "cosine" or "cosine_with_restarts" or "polynomial" or "constant" or "constant_with_warmup"
  warmup_steps = 1000,
  optim = "adafactor",
  weight_decay = 0.01,
  per_device_train_batch_size = 8,
  per_device_eval_batch_size = 8,
  gradient_accumulation_steps = 1,
  evaluation_strategy = "epoch",
  save_strategy = "epoch",
  #eval_steps = 100,
  predict_with_generate=True,
  generation_max_length = 128,
  save_steps = 100,
  logging_steps = 250,
  push_to_hub = False,
  fp16 = True,
  load_best_model_at_end=True,
  metric_for_best_model="rouge_1_mid_fmeasure",
  save_total_limit = 1, 
)

def compute_metrics(eval_pred):
    rouge_metric = datasets.load_metric("rouge")
    predictions, labels = eval_pred
    decoded_preds = mt5_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, mt5_tokenizer.pad_token_id)
    decoded_labels = mt5_tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=True)

    # Extract the mid F-measure for ROUGE-1, ROUGE-2, and ROUGE-L
    rouge_1_mid_fmeasure = rouge['rouge1'].mid.fmeasure
    rouge_2_mid_fmeasure = rouge['rouge2'].mid.fmeasure
    rouge_l_mid_fmeasure = rouge['rougeL'].mid.fmeasure

    return {
        "rouge_1_mid_fmeasure": rouge_1_mid_fmeasure,
        "rouge_2_mid_fmeasure": rouge_2_mid_fmeasure,
        "rouge_l_mid_fmeasure": rouge_l_mid_fmeasure
    }

print(tokenized_dataset["train"])
trainer = Seq2SeqTrainer(
  model = model,
  args = training_args,
  data_collator = data_collator,
  compute_metrics = compute_metrics,
  train_dataset = tokenized_dataset["train"],
  eval_dataset = tokenized_dataset["validation"],
  tokenizer = mt5_tokenizer,
)

trainer.train()


# save fine-tuned model in local
os.makedirs("./large_model", exist_ok=True)
if hasattr(trainer.model, "module"):
  trainer.model.module.save_pretrained("./large_model")
else:
  trainer.model.save_pretrained("./large_model")

