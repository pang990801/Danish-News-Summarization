import os
import numpy as np
import nltk
import torch
import datasets
from datasets import load_dataset, DatasetDict
import evaluate
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoConfig
)

model_name = "google/mt5-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Datasets
#test_dataset = load_dataset("ScandEval/nordjylland-news-summarization-mini", split="test")
test_dataset = load_dataset("alexandrainst/nordjylland-news-summarization", split="test")
split_dataset = DatasetDict({'test': test_dataset})

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization
def tokenize_data(data):
    input_feature = tokenizer(data["input_text"], truncation=True, max_length=1024)
    label = tokenizer(data["target_text"], truncation=True, max_length=128)
    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": label["input_ids"],
    }

tokenized_dataset = split_dataset.map(
    tokenize_data,
    remove_columns=['input_text', 'target_text', 'text_len', 'summary_len'],
    batched=True,
    batch_size=128
)

model = AutoModelForSeq2SeqLM.from_pretrained("small_model").to(device)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

training_args = Seq2SeqTrainingArguments(
    output_dir="mt5-summarize-large",
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    predict_with_generate=True,
    dataloader_drop_last=True,
)

# Compute Metrics
def compute_metrics(eval_pred):
    rouge_metric = datasets.load_metric("rouge")
    bert_metric = datasets.load_metric("bertscore")

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE scores
    rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=True)
    rouge_results = {f'rouge_{key}_mid_fmeasure': value.mid.fmeasure for key, value in rouge.items()}

    # Compute BERTScores
    bert_scores = bert_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        lang='da',
        model_type="xlm-roberta-large"
    )
    bertscore_results = {
        "bertscore_precision": np.mean(bert_scores["precision"]),
        "bertscore_recall": np.mean(bert_scores["recall"]),
        "bertscore_f1": np.mean(bert_scores["f1"])
    }

    # Combine metrics
    metrics = {**rouge_results, **bertscore_results}

    return metrics

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

metrics = trainer.evaluate()
print(metrics)

