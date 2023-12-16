import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import ctranslate2

nltk.download('punkt')

# Configuration settings
input_file_path = 'summarized_file_ml80_rp5.0.csv'
output_file_path = 'labeled_dataset_ml80_rp5.0.csv'
model_path = "models/opus-mt-en-da_ct2"
max_length = 512
text_max_length = int(max_length * 0.9)

df = pd.read_csv(input_file_path)
summaries = df['summary'].tolist()
bodys = df['body'].tolist()
ids = df['id'].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the translator and tokenizer
translator = ctranslate2.Translator(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-da", use_auth_token=False, src_lang="eng_Latn")


def translate(text, src_lang='eng_Latn', tgt_lang='dan_Latn', max_input_length=512):
    """Translate a list of texts using CTranslate2."""
    source_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence)) for sentence in text]
    target_prefixes = [[tgt_lang] for _ in source_tokens]
    results = translator.translate_batch(source_tokens, target_prefix=target_prefixes)
    translations = [tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0][1:])) for result in results]
    return translations


def split_into_sentences(article, text_max_length, tokenizer):
    """Split the article into lists of sentences each under max_length."""
    sentences = sent_tokenize(article)
    sentence_lists = []
    current_list = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=True))
        if current_length + sentence_length > text_max_length:
            sentence_lists.append(current_list)
            current_list = []
            current_length = 0
        current_list.append(sentence)
        current_length += sentence_length

    if current_list:
        sentence_lists.append(current_list)

    return sentence_lists


def translate_article(article, src_lang='eng_Latn', tgt_lang='dan_Latn', max_input_length=512, tokenizer=tokenizer):
    """Translate an entire article by breaking it into sentences."""
    sentence_lists = split_into_sentences(article, max_length, tokenizer)
    translated_article = ''

    for sentences in sentence_lists:
        translated_sentences = translate(sentences, src_lang, tgt_lang, max_input_length)
        translated_article += ' '.join(translated_sentences) + ' '

    return translated_article.strip()


translated_summaries = []

for summary in tqdm(summaries, desc="Translating Summary"):
    translated_summary = translate_article(summary)
    translated_summaries.append(translated_summary)

df = pd.DataFrame({
    'id': ids,
    'body': bodys,
    'summary': translated_summaries
})
df.to_csv(output_file_path, index=False, encoding='utf-8')
