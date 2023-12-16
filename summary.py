import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

input_file_path = "translated_articles.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "mrm8488/t5-base-finetuned-summarize-news"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
text_max_length = 512
# ml is max_length of summary, rp is repetition_penalty in model.generate()
output_file_path = 'summarized_file_ml80_rp5.0.csv'

df = pd.read_csv(input_file_path)[:10]
translated = df['translated']
bodys = df['body'].tolist()
ids = df['id'].tolist()

def summarize(text, max_length=80):
    """Summarize a given text."""
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,
                                   repetition_penalty=5.0, length_penalty=1.0, early_stopping=True)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds[0]


def split_long_sentence(sentence, max_length):
    """Split a long sentence into smaller parts."""
    words = word_tokenize(sentence)
    current_chunk, chunks, current_length = [], [], 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if word in [',', ';', ':'] and current_length < max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk, current_length = [], 0
        elif current_length >= max_length:
            last_word = current_chunk.pop()
            chunks.append(' '.join(current_chunk))
            current_chunk, current_length = [last_word], len(last_word) + 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def split_into_sentences(article, text_max_length, tokenizer):
    """Split article into sentences, ensuring each is under the max length."""
    raw_sentences = sent_tokenize(article)
    sentence_tokens_lengths = []
    # Ensure each sentence is under the max length
    for sentence in raw_sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        sentence_length = len(tokens)

        if sentence_length > text_max_length:
            # Split long sentence and update tokens lengths
            split_chunks = split_long_sentence(sentence, text_max_length)
            for chunk in split_chunks:
                chunk_length = len(tokenizer.encode(chunk, add_special_tokens=True))
                sentence_tokens_lengths.append((chunk, chunk_length))
        else:
            sentence_tokens_lengths.append((sentence, sentence_length))
    sentence_lists = []
    current_list = []
    current_length = 0
    for sentence, length in sentence_tokens_lengths:
        if current_length + length > text_max_length:
            sentence_lists.append(current_list)
            current_list = [sentence]
            current_length = length
        else:
            current_list.append(sentence)
            current_length += length
    if current_list:
        sentence_lists.append(current_list)
    return sentence_lists

def sentences_to_text(sentence_lists):
    """Convert list of sentences to text."""
    return [' '.join(sentences) for sentences in sentence_lists]

def split_article(article, text_max_length, tokenizer):
    """Split article into sub-articles, each fitting the max length constraint."""
    sentence_lists = split_into_sentences(article, text_max_length=text_max_length, tokenizer=tokenizer)
    sub_articles = sentences_to_text(sentence_lists)
    return sub_articles


summaries = []
for body in tqdm(translated, desc="Summarizing"):
    sub_articles = split_article(body, text_max_length=text_max_length, tokenizer=tokenizer)
    sub_summaries = []
    for sub_article in sub_articles:
        sub_summary = summarize(sub_article)
        sub_summaries.append(sub_summary)
    summary = " ".join(sub_summaries)
    summaries.append(summary)

df = pd.DataFrame({
    'id': ids,
    'body': bodys,
    'translated': translated,
    'summary': summaries,
})

df.to_csv(output_file_path, index=False, encoding='utf-8')

