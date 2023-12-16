import os
import numpy as np
import nltk
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import ctranslate2

nltk.download('punkt')

# Configuration settings
input_file_path = '10000_articles_without_linebreaks.csv'
model_path = "models/opus-mt-da-en_ct2"  # Path to the translation model
max_length = 512
text_max_length = int(max_length * 0.9)
output_file_path = 'translated_articles.csv'

df = pd.read_csv(input_file_path)
bodys = df['article text'].tolist()
ids = df['id'].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the translator and tokenizer
translator = ctranslate2.Translator(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-da-en", use_auth_token=False, src_lang="dan_Latn")


def translate(text, src_lang='dan_Latn', tgt_lang='eng_Latn', max_input_length=512):
    """Translate a list of texts using CTranslate2."""
    source_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence)) for sentence in text]
    target_prefixes = [[tgt_lang] for _ in source_tokens]
    results = translator.translate_batch(source_tokens, target_prefix=target_prefixes)
    translations = [tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0][1:])) for result in results]
    return translations


def split_long_sentence(sentence, max_length):
    """Split a long sentence into chunks smaller than max_length."""
    words = word_tokenize(sentence)
    current_chunk = []
    chunks = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if word in [',', ';', ':'] and current_length < max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        elif current_length >= max_length:
            last_word = current_chunk.pop()
            chunks.append(' '.join(current_chunk))
            current_chunk = [last_word]
            current_length = len(last_word) + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def split_into_sentences(article):
    """Split the article into lists of sentences."""
    raw_sentences = sent_tokenize(article)
    sentence_tokens_lengths = []

    for sentence in raw_sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        sentence_length = len(tokens)

        if sentence_length > text_max_length:
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


def translate_article(article):
    """Translate an entire article."""
    sentence_lists = split_into_sentences(article)
    translated_article = ''

    for sentences in sentence_lists:
        translated_sentences = translate(sentences)
        translated_article += ' '.join(translated_sentences) + ' '

    return translated_article.strip()


# Main processing loop
translated_bodys = []
for article in tqdm(bodys, desc="Translating Articles"):
    translated_paragraphs = translate_article(article)
    translated_bodys.append(translated_paragraphs)

# Save the results to DataFrame and export to CSV
df = pd.DataFrame({
    'id': ids,
    'body': bodys,
    'translated': translated_bodys
})
df.to_csv(output_file_path, index=False, encoding='utf-8')
