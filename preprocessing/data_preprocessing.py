from langdetect import detect 
from langdetect.lang_detect_exception import LangDetectException
import re
import ftfy
import pandas as pd
from tqdm import tqdm
from nltk import tokenize
import fasttext
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

df_og = pd.read_csv('dataset_cleand-1-1.csv')
df = df_og.copy()
print("Original DataFrame:")
print(df.head(20))

def remove_symbols(text):
    return re.sub(r'[^A-Za-z0-9\s.,:;!?\'"/\-]', '', text)

def add_remove_space_near_puncuation(text):
    text = re.sub(r'\s+([.!?])', r'\1', text) 
    text = re.sub(r'([.!?])([a-zA-Z])', r'\1 \2', text) 
    return text

def remove_short_senteces(text):
    if not isinstance(text, str) or text is None:
        return ''
    sentences = tokenize.sent_tokenize(text)
    new_text = ''
    for sentence in sentences:
        if len(sentence.split()) >= 3:
            new_text += sentence + ' '
    return new_text.strip()


def filter_non_english_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    english_sentences = []
    for sentence in sentences:
        try:
            if detect(sentence) == 'en':  
                english_sentences.append(sentence)
        except LangDetectException:
            pass
    return ' '.join(english_sentences)


def additional_fasttext_language_detection(text):
    english_sentences = []
    sentences = re.split(r'(?<=[.!?]) +', text)
    for sentence in sentences:
        prediction = model.predict(sentence, k=1)
        predicted_language = prediction[0][0].split('__label__')[1]
        confidence_score = prediction[1][0]
        if predicted_language == 'eng_Latn' and confidence_score >= 0.99:
            english_sentences.append(sentence)

    return ' '.join(english_sentences)


def remove_paragraph_space(text):
    return re.sub(r'\n+', '', text.strip())


def remove_emoticons(text):
    emoticon_pattern = re.compile(
        r'(:\)|:\(|\(:|\)\:|:\-\)|:\-\(|\^\^|<3|:P|:D|;\)|:-D|:O|:-O|:\')',
        flags=re.UNICODE
    )
    def replace_emoticon(match):
        end_pos = match.end()
        if end_pos < len(text) and text[end_pos] in ".!? ":
            return ''  
        else:
            return '.'  
    return emoticon_pattern.sub(replace_emoticon, text)

df = df.drop_duplicates(subset=['text'], keep='first')

tqdm.pandas(desc="Fixing text encoding")
df['text'] = df['text'].progress_apply(ftfy.fix_text)

tqdm.pandas(desc="Removing emoticons")
df['text'] = df['text'].progress_apply(remove_emoticons)

tqdm.pandas(desc="Removing symbols")
df['text'] = df['text'].progress_apply(remove_symbols)

tqdm.pandas(desc="Adding/removing spaces near punctuations")
df['text'] = df['text'].progress_apply(add_remove_space_near_puncuation)

tqdm.pandas(desc="Removing paragraph spaces")
df['text'] = df['text'].progress_apply(remove_paragraph_space)

tqdm.pandas(desc="Filtering for non-English sentences")
df['text'] = df['text'].progress_apply(filter_non_english_sentences)
df = df[df['text'].str.strip().astype(bool)].reset_index(drop=True)

tqdm.pandas(desc="Additional filtering with fasttext model")
df['text'] = df['text'].progress_apply(additional_fasttext_language_detection)
df = df[df['text'].str.strip().astype(bool)].reset_index(drop=True)

tqdm.pandas(desc="Removing short sentences")
df['text'] = df['text'].progress_apply(remove_short_senteces)
df['text'].replace('', pd.NA, inplace=True)
df = df.dropna(subset=['text']).reset_index(drop=True)

print("Processed DataFrame:")
print(df.head(20))

df.to_csv('preprocessed_dataset.csv', index=False)